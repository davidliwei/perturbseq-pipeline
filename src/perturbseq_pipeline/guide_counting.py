"""Streaming, scaffold-aware guide quantification from feature-barcode FASTQs.

Read structure (10x 5' feature barcode, defaults in ``guides.fastq``)::

    R1 = [16 nt cell barcode][12 nt UMI][TSO][0-n G][20 nt protospacer][scaffold ...]

The protospacer is located *relative to the scaffold anchor* (the 20 nt that
end where the anchor starts), which makes the parser robust to the variable
run of non-templated G bases after the TSO. Anchors are configured per
scaffold class (e.g. ``A: GTTTAAGAGCTA``, ``C: GTTTCAGAGCTA``) so each read
also reports which scaffold it carries.

Matching is exact by default. Two explicit, configurable relaxations exist:

``position_shift``
    retry the *exact* sequence match at +/- n positions (indels near the TSO
    or anchor); still an exact sequence match.
``max_mismatches = 1``
    accept a single substitution when the variant is unambiguous.

Counting is per (cell barcode, guide, UMI): reads are collapsed to unique
UMIs so the matrix holds guide UMI counts, not read depth. Cellular counts
are restricted to the paired GEX barcode universe; reads from other barcodes
are tallied but not stored per UMI, which keeps memory bounded.

Files are processed in parallel (one worker process per FASTQ file), each
streaming through ``pigz``/``gzip`` without ever holding a file in memory.
"""

from __future__ import annotations

import collections
import gzip
import json
import logging
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import Config, GuideFastqConfig

logger = logging.getLogger(__name__)

GUIDE_FEATURE_TYPE = "CRISPR Guide Capture"

_UMI_TABLE = bytes.maketrans(b"ACGT", b"0123")


# Specification


@dataclass(frozen=True)
class GuideReadSpec:
    """Immutable, picklable read-structure description handed to workers."""

    barcode_length: int
    umi_length: int
    protospacer_length: int
    scaffold_names: Tuple[str, ...]
    scaffold_anchors: Tuple[str, ...]
    anchor_search_start: int
    position_shift: int
    max_mismatches: int
    tso: Optional[str]

    @classmethod
    def from_config(cls, fq: GuideFastqConfig) -> "GuideReadSpec":
        names = tuple(str(k) for k in fq.scaffolds.keys())
        anchors = tuple(str(v).upper() for v in fq.scaffolds.values())
        return cls(
            barcode_length=int(fq.barcode_length),
            umi_length=int(fq.umi_length),
            protospacer_length=int(fq.protospacer_length),
            scaffold_names=names,
            scaffold_anchors=anchors,
            anchor_search_start=int(fq.anchor_search_start),
            position_shift=int(fq.position_shift),
            max_mismatches=int(fq.max_mismatches),
            tso=str(fq.tso).upper() if fq.tso else None,
        )

    def anchor_regex(self) -> "re.Pattern[bytes]":
        parts = [f"({re.escape(a)})" for a in self.scaffold_anchors]
        return re.compile("|".join(parts).encode())

    @property
    def umi_bits(self) -> int:
        return 2 * self.umi_length


def build_protospacer_index(
    protospacers: Sequence[str], max_mismatches: int = 0
) -> Tuple[Dict[bytes, int], Dict[bytes, int]]:
    """Exact index (first occurrence wins) and optional unambiguous 1-mismatch index."""
    exact: Dict[bytes, int] = {}
    for i, p in enumerate(protospacers):
        key = str(p).upper().encode()
        exact.setdefault(key, i)
    mm: Dict[bytes, int] = {}
    if max_mismatches >= 1:
        ambiguous: set = set()
        for key, i in exact.items():
            for pos in range(len(key)):
                for base in b"ACGT":
                    if key[pos] == base:
                        continue
                    var = key[:pos] + bytes([base]) + key[pos + 1 :]
                    if var in exact:
                        continue
                    if var in mm and mm[var] != i:
                        ambiguous.add(var)
                    else:
                        mm[var] = i
        for var in ambiguous:
            mm.pop(var, None)
    return exact, mm


# Streaming


def _choose_decompressor(prefer: str = "auto") -> Optional[str]:
    if prefer == "python":
        return None
    for tool in (prefer,) if prefer in ("pigz", "gzip") else ("pigz", "gzip"):
        exe = shutil.which(tool)
        if exe:
            return exe
    return None


def open_sequence_stream(path: str | Path, decompressor: str = "auto"):
    """Yield the sequence line (bytes, newline-stripped) of every FASTQ record.

    Returns ``(iterator, closer)``; call ``closer()`` when done.
    """
    path = Path(path)
    if path.suffix == ".gz":
        exe = _choose_decompressor(decompressor)
        if exe:
            args = [exe, "-dc"]
            if os.path.basename(exe) == "pigz":
                args += ["-p", "2"]
            proc = subprocess.Popen(args + [str(path)], stdout=subprocess.PIPE, bufsize=1 << 22)
            handle = proc.stdout

            def closer():
                try:
                    handle.close()
                finally:
                    proc.wait()
        else:
            handle = gzip.open(path, "rb")
            proc = None

            def closer():
                handle.close()
    else:
        handle = open(path, "rb", buffering=1 << 22)

        def closer():
            handle.close()

    seqs = (line.rstrip(b"\r\n") for line in islice(handle, 1, None, 4))
    return seqs, closer


def _resolve_guide(seq: bytes, p: int, L: int, exact: Dict[bytes, int], mm: Dict[bytes, int], shift: int):
    """Return (guide_index, how) or (None, None)."""
    proto = seq[p - L : p]
    g = exact.get(proto)
    if g is not None:
        return g, 0
    if mm:
        g = mm.get(proto)
        if g is not None:
            return g, 2
    for sh in range(1, shift + 1):
        for s in (-sh, sh):
            start = p - L + s
            if start < 0:
                continue
            cand = seq[start : p + s]
            g = exact.get(cand)
            if g is not None:
                return g, 1
            if mm:
                g = mm.get(cand)
                if g is not None:
                    return g, 2
    return None, None


def count_fastq_file(
    path: str,
    spec: GuideReadSpec,
    exact_index: Dict[bytes, int],
    mm_index: Dict[bytes, int],
    barcode_index: Dict[bytes, int],
    n_guides: int,
    max_reads: Optional[int] = None,
    chunk_size: int = 2_000_000,
    unmatched_sample_rate: int = 50,
    decompressor: str = "auto",
    split_by_scaffold: bool = False,
) -> Dict[str, object]:
    """Stream one FASTQ file and return compact per-file results.

    Returns a dict with ``codes`` (unique int64 (cell, feature, UMI) codes for
    cells in the barcode universe), ``guide_reads`` (matched reads per feature
    over all barcodes), ``guide_scaffold_reads`` (features x scaffolds),
    ``unmatched`` (sampled Counter of unmatched protospacers) and ``stats``.

    With ``split_by_scaffold`` a feature is a (designed protospacer, scaffold
    class) combination indexed ``guide * n_scaffolds + scaffold``; otherwise a
    feature is the designed protospacer regardless of the scaffold it carries.
    """
    t0 = time.time()
    anchor_re = spec.anchor_regex()
    B, U, L = spec.barcode_length, spec.umi_length, spec.protospacer_length
    umi_end = B + U
    umi_bits = spec.umi_bits
    search_start = spec.anchor_search_start
    shift = spec.position_shift
    n_scaf = len(spec.scaffold_anchors)
    n_features = n_guides * n_scaf if split_by_scaffold else n_guides
    tso = spec.tso.encode() if spec.tso else None
    tso_lo, tso_hi = umi_end - 4, umi_end + (len(tso) if tso else 0) + 8
    guide_reads = [0] * n_features
    guide_scaf = [[0] * n_scaf for _ in range(n_features)]
    scaf_reads = [0] * n_scaf
    unmatched: collections.Counter = collections.Counter()
    codes: List[int] = []
    chunks: List[np.ndarray] = []
    n = n_anchor = n_tso = n_matched = n_shift = n_mm = n_mm_pos1 = 0
    n_bc_hit = n_bc_miss = n_bad_umi = n_unmatched_sampled_total = 0
    # designed protospacer per guide index, to classify 1-mismatch rescues
    designed_by_index: Dict[int, bytes] = {i: k for k, i in exact_index.items()}

    def flush():
        if codes:
            arr = np.unique(np.asarray(codes, dtype=np.int64))
            chunks.append(arr)
            codes.clear()
            if len(chunks) > 8:
                merged = np.unique(np.concatenate(chunks))
                chunks.clear()
                chunks.append(merged)

    seqs, closer = open_sequence_stream(path, decompressor)
    try:
        if max_reads is not None:
            seqs = islice(seqs, max_reads)
        for seq in seqs:
            n += 1
            if tso is not None and seq.find(tso, tso_lo, tso_hi) >= 0:
                n_tso += 1
            m = anchor_re.search(seq, search_start)
            if m is None:
                continue
            n_anchor += 1
            p = m.start()
            scaf = m.lastindex - 1
            scaf_reads[scaf] += 1
            if p < L:
                continue
            g, how = _resolve_guide(seq, p, L, exact_index, mm_index, shift)
            if g is None:
                n_unmatched_sampled_total += 1
                if (n_unmatched_sampled_total - 1) % unmatched_sample_rate == 0:
                    unmatched[seq[p - L : p]] += 1
                continue
            n_matched += 1
            if how == 1:
                n_shift += 1
            elif how == 2:
                n_mm += 1
                # 5' (position 1) substitution, e.g. the U6 +1 G: same 19-nt suffix
                if seq[p - L + 1 : p] == designed_by_index[g][1:]:
                    n_mm_pos1 += 1
            f = g * n_scaf + scaf if split_by_scaffold else g
            guide_reads[f] += 1
            guide_scaf[f][scaf] += 1
            c = barcode_index.get(seq[:B])
            if c is None:
                n_bc_miss += 1
                continue
            n_bc_hit += 1
            try:
                u = int(seq[B:umi_end].translate(_UMI_TABLE), 4)
            except ValueError:
                n_bad_umi += 1
                continue
            codes.append(((c * n_features + f) << umi_bits) | u)
            if len(codes) >= chunk_size:
                flush()
    finally:
        closer()
    flush()
    all_codes = np.unique(np.concatenate(chunks)) if chunks else np.zeros(0, dtype=np.int64)
    elapsed = time.time() - t0
    stats = {
        "file": str(path),
        "reads_total": n,
        "reads_with_tso": n_tso,
        "reads_with_scaffold_anchor": n_anchor,
        "reads_spacer_matched": n_matched,
        "reads_spacer_matched_via_shift": n_shift,
        "reads_spacer_matched_via_mismatch": n_mm,
        "reads_spacer_matched_via_mismatch_pos1": n_mm_pos1,
        "reads_spacer_unmatched": n_anchor - n_matched,
        "reads_matched_barcode_in_gex": n_bc_hit,
        "reads_matched_barcode_not_in_gex": n_bc_miss,
        "reads_invalid_umi": n_bad_umi,
        "unique_cell_guide_umis": int(all_codes.size),
        "elapsed_seconds": round(elapsed, 1),
        "reads_per_second": round(n / elapsed) if elapsed > 0 else None,
    }
    for name, k in zip(spec.scaffold_names, scaf_reads):
        stats[f"reads_scaffold_{name}"] = k
    return {
        "codes": all_codes,
        "guide_reads": np.asarray(guide_reads, dtype=np.int64),
        "guide_scaffold_reads": np.asarray(guide_scaf, dtype=np.int64).reshape(n_features, n_scaf),
        "unmatched": unmatched,
        "stats": stats,
    }


# Results


@dataclass
class GuideCountResult:
    """Guide UMI counts for one sample over the paired GEX barcode universe."""

    sample_id: str
    #: Designed guide ids (matrix columns), in reference order.
    guide_ids: List[str]
    #: Bare cell barcodes (matrix rows), in GEX obs order.
    cell_barcodes: List[str]
    #: cells x guides UMI counts (CSR, int).
    counts: sp.csr_matrix
    #: Matched reads per guide across all barcodes.
    guide_reads: np.ndarray
    #: Matched reads per guide split by scaffold class (guides x scaffolds).
    guide_scaffold_reads: np.ndarray
    scaffold_names: List[str]
    stats: Dict[str, object] = field(default_factory=dict)
    per_file: List[Dict[str, object]] = field(default_factory=list)
    unmatched_top: Optional[pd.DataFrame] = None
    source: str = "fastq"
    #: Feature-level design frame when features are (spacer x scaffold)
    #: combinations (``guides.fastq.scaffold_specific_features``); ``None``
    #: when features are the designed guides themselves.
    feature_design: Optional[pd.DataFrame] = None

    @property
    def n_cells(self) -> int:
        return self.counts.shape[0]

    @property
    def n_guides(self) -> int:
        return self.counts.shape[1]

    def guide_umis(self) -> np.ndarray:
        return np.asarray(self.counts.sum(axis=0)).ravel()

    def guide_positive_cells(self, threshold: int = 1) -> np.ndarray:
        return np.asarray((self.counts >= threshold).sum(axis=0)).ravel()


def _codes_to_matrix(codes: np.ndarray, n_cells: int, n_guides: int, umi_bits: int) -> sp.csr_matrix:
    if codes.size == 0:
        return sp.csr_matrix((n_cells, n_guides), dtype=np.int32)
    cg = codes >> umi_bits
    keys, umis = np.unique(cg, return_counts=True)
    rows = (keys // n_guides).astype(np.int64)
    cols = (keys % n_guides).astype(np.int64)
    mat = sp.coo_matrix((umis.astype(np.int32), (rows, cols)), shape=(n_cells, n_guides))
    return mat.tocsr()


def find_guide_fastqs(directory: str | Path, pattern: str) -> List[str]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"guide_fastq_dir does not exist: {directory}")
    files = sorted(str(p) for p in directory.rglob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No guide FASTQ matched {pattern!r} under {directory}")
    return files


@dataclass
class GuideCountJob:
    sample_id: str
    fastq_files: List[str]
    cell_barcodes: List[str]


FEATURE_SEP = ":"


def expand_design_by_scaffold(design: pd.DataFrame, scaffold_names: Sequence[str]) -> pd.DataFrame:
    """One feature row per (designed guide x scaffold class), ids ``<guide_id>:<scaffold>``.

    Row order is ``guide-major`` (guide 0 x every scaffold, guide 1 x ...), matching
    the feature index ``guide * n_scaffolds + scaffold`` used by the counter. The
    designed guide's own columns are kept; ``design_guide_id`` holds the original
    id, ``scaffold`` the class of the feature and ``designed_slot`` whether the
    design table declared that class for the guide (``True`` when the design has
    no scaffold information at all).
    """
    names = [str(n) for n in scaffold_names]
    rows = []
    has_scaf = "scaffold" in design.columns and (design["scaffold"].astype(str) != "unknown").any()
    for _, r in design.reset_index(drop=True).iterrows():
        declared = {x.strip() for x in str(r.get("scaffold", "unknown")).split(";")} if has_scaf else set()
        for cls in names:
            row = r.to_dict()
            row["design_guide_id"] = r["guide_id"]
            row["guide_id"] = f"{r['guide_id']}{FEATURE_SEP}{cls}"
            row["scaffold"] = cls
            row["scaffold_source"] = "feature_split_by_read_anchor"
            row["designed_slot"] = (cls in declared) if has_scaf else True
            rows.append(row)
    out = pd.DataFrame(rows)
    out["design_index"] = np.arange(len(out), dtype=int)
    return out


def count_guides(
    jobs: Sequence[GuideCountJob], design: pd.DataFrame, cfg: Config, n_workers: Optional[int] = None
) -> Dict[str, GuideCountResult]:
    """Count guide UMIs for several samples with one shared worker pool."""
    fq = cfg.guides.fastq
    spec = GuideReadSpec.from_config(fq)
    protospacers = design["protospacer"].astype(str).tolist()
    n_guides = len(protospacers)
    split = bool(fq.scaffold_specific_features)
    feature_design = expand_design_by_scaffold(design, spec.scaffold_names) if split else None
    guide_ids = (feature_design if split else design)["guide_id"].astype(str).tolist()
    n_features = len(guide_ids)
    exact, mm = build_protospacer_index(protospacers, spec.max_mismatches)
    for job in jobs:
        top = (len(job.cell_barcodes) * n_features) << spec.umi_bits
        if top >= 2**62:
            raise ValueError(
                f"{job.sample_id}: cells x guides x UMI space does not fit the int64 code; "
                "reduce umi_length or split the sample"
            )
    tasks: List[Tuple[str, str]] = [(j.sample_id, f) for j in jobs for f in j.fastq_files]
    if not tasks:
        return {}
    bc_indexes = {j.sample_id: {b.encode(): i for i, b in enumerate(j.cell_barcodes)} for j in jobs}
    for j in jobs:
        if len(bc_indexes[j.sample_id]) != len(j.cell_barcodes):
            raise ValueError(f"{j.sample_id}: duplicate bare barcodes in the GEX universe")
    workers = n_workers or fq.n_workers or min(len(tasks), max(1, os.cpu_count() or 1))
    workers = max(1, min(workers, len(tasks)))
    logger.info(
        "Guide counting: %d FASTQ file(s) across %d sample(s) with %d worker(s); "
        "%d designed guides -> %d features (%s), %d scaffold class(es) %s, position_shift=%d, max_mismatches=%d",
        len(tasks),
        len(jobs),
        workers,
        n_guides,
        n_features,
        "spacer x scaffold class" if split else "spacer only",
        len(spec.scaffold_names),
        list(spec.scaffold_names),
        spec.position_shift,
        spec.max_mismatches,
    )
    per_file: Dict[str, List[Dict[str, object]]] = {j.sample_id: [] for j in jobs}
    kwargs = dict(
        max_reads=fq.max_reads,
        chunk_size=fq.chunk_size,
        unmatched_sample_rate=max(1, fq.unmatched_sample_rate),
        decompressor="auto",
        split_by_scaffold=split,
    )
    t0 = time.time()
    if workers == 1:
        for sid, f in tasks:
            res = count_fastq_file(f, spec, exact, mm, bc_indexes[sid], n_guides, **kwargs)
            per_file[sid].append(res)
            logger.info(
                "  %s: %s done (%s reads, %.0f s)",
                sid,
                Path(f).name,
                f"{res['stats']['reads_total']:,}",
                res["stats"]["elapsed_seconds"],
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(count_fastq_file, f, spec, exact, mm, bc_indexes[sid], n_guides, **kwargs): (sid, f)
                for sid, f in tasks
            }
            for fut in as_completed(futures):
                sid, f = futures[fut]
                res = fut.result()
                per_file[sid].append(res)
                logger.info(
                    "  %s: %s done (%s reads, %.0f s)",
                    sid,
                    Path(f).name,
                    f"{res['stats']['reads_total']:,}",
                    res["stats"]["elapsed_seconds"],
                )
    logger.info("Guide counting finished in %.0f s", time.time() - t0)
    out: Dict[str, GuideCountResult] = {}
    for job in jobs:
        results = sorted(per_file[job.sample_id], key=lambda r: r["stats"]["file"])
        codes = np.unique(np.concatenate([r["codes"] for r in results])) if results else np.zeros(0, dtype=np.int64)
        counts = _codes_to_matrix(codes, len(job.cell_barcodes), n_features, spec.umi_bits)
        guide_reads = (
            np.sum([r["guide_reads"] for r in results], axis=0) if results else np.zeros(n_features, dtype=np.int64)
        )
        gsr = (
            np.sum([r["guide_scaffold_reads"] for r in results], axis=0)
            if results
            else np.zeros((n_features, len(spec.scaffold_names)), dtype=np.int64)
        )
        unmatched: collections.Counter = collections.Counter()
        for r in results:
            unmatched.update(r["unmatched"])
        stats = _aggregate_stats([r["stats"] for r in results], spec)
        stats["n_files"] = len(results)
        stats["unique_cell_guide_umis"] = int(codes.size)
        stats["total_guide_umis_in_matrix"] = int(counts.sum())
        stats["guides_detected_any_umi"] = int((counts.sum(axis=0) > 0).sum())
        stats["guides_designed"] = n_guides
        stats["features"] = n_features
        stats["scaffold_specific_features"] = split
        stats["cells_in_gex_universe"] = len(job.cell_barcodes)
        stats["cells_with_any_guide_umi"] = int((counts.sum(axis=1) > 0).sum())
        top = pd.DataFrame(
            [(k.decode(), v * kwargs["unmatched_sample_rate"]) for k, v in unmatched.most_common(fq.unmatched_top_n)],
            columns=["protospacer", "estimated_reads"],
        )
        out[job.sample_id] = GuideCountResult(
            sample_id=job.sample_id,
            guide_ids=guide_ids,
            cell_barcodes=list(job.cell_barcodes),
            counts=counts,
            guide_reads=np.asarray(guide_reads, dtype=np.int64),
            guide_scaffold_reads=np.asarray(gsr, dtype=np.int64).reshape(n_features, len(spec.scaffold_names)),
            scaffold_names=list(spec.scaffold_names),
            stats=stats,
            per_file=[r["stats"] for r in results],
            unmatched_top=top,
            source="fastq",
            feature_design=feature_design,
        )
        logger.info(
            "%s: %s reads, %.1f%% with scaffold anchor, %.1f%% spacer-matched, %.1f%% of matched "
            "reads in GEX barcodes; %d/%d features observed, %d unique cell-guide UMIs",
            job.sample_id,
            f"{stats['reads_total']:,}",
            100 * stats["frac_reads_with_scaffold_anchor"],
            100 * stats["frac_reads_spacer_matched"],
            100 * stats["frac_matched_reads_in_gex_barcodes"],
            stats["guides_detected_any_umi"],
            n_features,
            codes.size,
        )
    return out


def _aggregate_stats(stats: List[Dict[str, object]], spec: GuideReadSpec) -> Dict[str, object]:
    agg: Dict[str, object] = {}
    sum_keys = [
        "reads_total",
        "reads_with_tso",
        "reads_with_scaffold_anchor",
        "reads_spacer_matched",
        "reads_spacer_matched_via_shift",
        "reads_spacer_matched_via_mismatch",
        "reads_spacer_matched_via_mismatch_pos1",
        "reads_spacer_unmatched",
        "reads_matched_barcode_in_gex",
        "reads_matched_barcode_not_in_gex",
        "reads_invalid_umi",
    ] + [f"reads_scaffold_{n}" for n in spec.scaffold_names]
    for k in sum_keys:
        agg[k] = int(sum(int(s.get(k, 0)) for s in stats))
    agg["elapsed_seconds_sum"] = float(sum(float(s.get("elapsed_seconds", 0)) for s in stats))
    tot = agg["reads_total"] or 1
    agg["frac_reads_with_tso"] = agg["reads_with_tso"] / tot
    agg["frac_reads_with_scaffold_anchor"] = agg["reads_with_scaffold_anchor"] / tot
    agg["frac_reads_spacer_matched"] = agg["reads_spacer_matched"] / tot
    matched = agg["reads_spacer_matched"] or 1
    agg["frac_matched_reads_in_gex_barcodes"] = agg["reads_matched_barcode_in_gex"] / matched
    anchored = agg["reads_with_scaffold_anchor"] or 1
    agg["frac_anchored_reads_spacer_matched"] = agg["reads_spacer_matched"] / anchored
    for n in spec.scaffold_names:
        agg[f"frac_anchored_reads_scaffold_{n}"] = agg[f"reads_scaffold_{n}"] / anchored
    return agg


# Pre-computed guide matrices


def guide_counts_from_matrix(
    sample_id: str, path: str | Path, cell_barcodes: Sequence[str], design: Optional[pd.DataFrame], cfg: Config
) -> Tuple[GuideCountResult, pd.DataFrame]:
    """Build a :class:`GuideCountResult` from a 10x guide matrix (h5 or MTX).

    When a design table is given, matrix features are matched to designed
    guides by ``guide_id`` (then by protospacer if the matrix carries a
    ``sequence`` column); designed guides absent from the matrix get zero
    columns so the reference stays complete. Without a design table the
    matrix features *are* the reference (scaffold unknown).
    Returns ``(result, design_used)``.
    """
    from .io import read_10x_guide_features, strip_barcode_suffix

    gad = read_10x_guide_features(
        path,
        sample_id,
        guide_feature_types=cfg.input.guide_feature_types,
        feature_type_column=cfg.input.feature_type_column,
    )
    bare = strip_barcode_suffix(gad.obs_names, cfg.guides.fastq.barcode_suffix_regex)
    pos = {b: i for i, b in enumerate(bare)}
    rows = np.array([pos.get(b, -1) for b in cell_barcodes])
    hit = rows >= 0
    if hit.sum() == 0:
        raise ValueError(f"{sample_id}: no overlap between guide matrix barcodes and GEX barcodes ({path})")
    feat_ids = gad.var_names.astype(str).to_numpy()
    if design is None:
        design = pd.DataFrame(
            {
                "guide_id": feat_ids,
                "protospacer": gad.var["sequence"].astype(str).to_numpy()
                if "sequence" in gad.var
                else [""] * len(feat_ids),
                "target_raw": gad.var["guide_symbol"].astype(str).to_numpy() if "guide_symbol" in gad.var else feat_ids,
                "is_control": False,
                "scaffold": "unknown",
                "scaffold_source": "unspecified",
                "design_index": np.arange(len(feat_ids)),
            }
        )
        from .guide_design import is_control_label, CONTROL_TARGET_LABEL

        design["is_control"] = is_control_label(design["target_raw"].tolist(), cfg.guides.ntc_patterns)
        design["target"] = np.where(design["is_control"], CONTROL_TARGET_LABEL, design["target_raw"])
    col_of = {g: i for i, g in enumerate(design["guide_id"].astype(str))}
    if "sequence" in gad.var.columns:
        seq_of = {p: i for i, p in enumerate(design["protospacer"].astype(str).str.upper())}
    else:
        seq_of = {}
    cols = np.full(len(feat_ids), -1)
    for j, fid in enumerate(feat_ids):
        if fid in col_of:
            cols[j] = col_of[fid]
        elif seq_of:
            cols[j] = seq_of.get(str(gad.var["sequence"].iloc[j]).upper(), -1)
    unmatched_features = feat_ids[cols < 0].tolist()
    if unmatched_features:
        logger.warning(
            "%s: %d guide-matrix features not in the design reference (ignored): %s",
            sample_id,
            len(unmatched_features),
            unmatched_features[:5],
        )
    X = gad.X.tocsr()
    sub = X[rows[hit]][:, np.where(cols >= 0)[0]]
    counts_csr = sp.csr_matrix((len(cell_barcodes), len(design)), dtype=np.int32)
    coo = sub.tocoo()
    r = np.where(hit)[0][coo.row]
    c = cols[cols >= 0][coo.col]
    counts_csr = sp.coo_matrix((coo.data.astype(np.int32), (r, c)), shape=counts_csr.shape).tocsr()
    n_scaf = len(cfg.guides.fastq.scaffolds)
    res = GuideCountResult(
        sample_id=sample_id,
        guide_ids=design["guide_id"].astype(str).tolist(),
        cell_barcodes=list(cell_barcodes),
        counts=counts_csr,
        guide_reads=np.zeros(len(design), dtype=np.int64),
        guide_scaffold_reads=np.zeros((len(design), n_scaf), dtype=np.int64),
        scaffold_names=[str(k) for k in cfg.guides.fastq.scaffolds],
        stats={
            "source": str(path),
            "matrix_barcodes": int(gad.n_obs),
            "matrix_features": int(gad.n_vars),
            "cells_in_gex_universe": len(cell_barcodes),
            "cells_matched_in_matrix": int(hit.sum()),
            "features_not_in_design": len(unmatched_features),
            "total_guide_umis_in_matrix": int(counts_csr.sum()),
            "guides_detected_any_umi": int((counts_csr.sum(axis=0) > 0).sum()),
            "guides_designed": len(design),
        },
        source="matrix",
    )
    return res, design


# Writing


def write_guide_counts(result: GuideCountResult, design: pd.DataFrame, outdir: Path) -> Dict[str, Path]:
    """Write a 10x-style MTX bundle plus summaries for one sample."""
    import scipy.io

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if result.feature_design is not None and len(result.feature_design) == result.n_guides:
        design = result.feature_design
    if len(design) != result.n_guides:
        raise ValueError(
            f"{result.sample_id}: design has {len(design)} rows but the count matrix has {result.n_guides} features"
        )
    paths: Dict[str, Path] = {}
    mtx = outdir / "matrix.mtx.gz"
    with gzip.open(mtx, "wb") as fh:
        scipy.io.mmwrite(fh, result.counts.T.tocoo(), field="integer")
    paths["matrix"] = mtx
    bc = outdir / "barcodes.tsv.gz"
    with gzip.open(bc, "wt") as fh:
        fh.write("\n".join(result.cell_barcodes) + "\n")
    paths["barcodes"] = bc
    feats = outdir / "features.tsv.gz"
    targets = design["target_raw"].astype(str).tolist() if "target_raw" in design else result.guide_ids
    with gzip.open(feats, "wt") as fh:
        for gid, tgt in zip(result.guide_ids, targets):
            fh.write(f"{gid}\t{tgt}\t{GUIDE_FEATURE_TYPE}\n")
    paths["features"] = feats
    summary = design[
        [
            c
            for c in (
                "guide_id",
                "design_guide_id",
                "protospacer",
                "target",
                "target_raw",
                "is_control",
                "scaffold",
                "scaffold_source",
                "designed_slot",
            )
            if c in design.columns
        ]
    ].copy()
    summary["reads_matched"] = result.guide_reads
    summary["umis_in_cells"] = result.guide_umis()
    summary["cells_positive_any_umi"] = result.guide_positive_cells(1)
    for j, name in enumerate(result.scaffold_names):
        summary[f"reads_scaffold_{name}"] = result.guide_scaffold_reads[:, j]
    gs = outdir / f"{result.sample_id}_guide_summary.tsv"
    summary.to_csv(gs, sep="\t", index=False)
    paths["guide_summary"] = gs
    st = outdir / f"{result.sample_id}_guide_counting_stats.json"
    with open(st, "w") as fh:
        json.dump({"sample": result.stats, "per_file": result.per_file}, fh, indent=2, default=str)
    paths["stats"] = st
    if result.unmatched_top is not None and len(result.unmatched_top):
        um = outdir / f"{result.sample_id}_unmatched_protospacers.tsv"
        result.unmatched_top.to_csv(um, sep="\t", index=False)
        paths["unmatched"] = um
    return paths


def read_guide_counts(outdir: Path, sample_id: str) -> Tuple[sp.csr_matrix, List[str], List[str]]:
    """Read back an MTX bundle written by :func:`write_guide_counts`."""
    import scipy.io

    outdir = Path(outdir)
    with gzip.open(outdir / "matrix.mtx.gz", "rb") as fh:
        mat = scipy.io.mmread(fh).T.tocsr().astype(np.int32)
    with gzip.open(outdir / "barcodes.tsv.gz", "rt") as fh:
        barcodes = [l.strip() for l in fh if l.strip()]
    with gzip.open(outdir / "features.tsv.gz", "rt") as fh:
        guide_ids = [l.split("\t")[0] for l in fh if l.strip()]
    return mat, barcodes, guide_ids
