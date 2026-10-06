"""Synthetic multi-well 10x dataset for the basic QC stage tests.

Writes, per GEM well:

* a Cell Ranger-style ``filtered_feature_bc_matrix.h5`` (v3 layout);
* two gzipped guide "R1" FASTQ files with the 10x 5' feature-barcode read
  structure ``[16 bc][12 UMI][TSO][0-4 G][20 spacer][scaffold ...]``.

The design workbook mimics ``design_out_v2.xlsx`` (columns ``gene``, ``seq``,
``On-Target Efficacy Score``, ``BeginsWithG``) and includes ``NO-TARGET``
controls plus two guides that are designed but never sequenced.

Ground truth returned alongside the files: expected guide UMI counts per
cell, which cells were built as expression doublets, which carry a guide
multiplet, and which have no guide.
"""

from __future__ import annotations

import gzip
import random
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

TSO = "TTTCTTATATGGG"
SCAFFOLDS = {
    "A": "GTTTAAGAGCTAAGCTGGAAACAGCATAGCAAGTTTAAATAAGGCTAGTCCGTTATCAACTTGAGATCGGAAGAGCACACGTCTGAAC",
    "C": "GTTTCAGAGCTAAGCACAAGAGTGCATAGCAAGTTGAAATAAGGCTAGTCCGTTATCAACTTGAGATCGGAAGAGCACACGTCTGAAC",
}
TARGETS = ["GENEA", "GENEB", "GENEC", "GENED"]
N_GENES = 300


def _rand_seq(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def make_design(path: Path, seed: int = 0) -> pd.DataFrame:
    """Design workbook: 4 targets x (1 A + 1 C) + 4 NO-TARGET (2 A, 2 C) + 2 unsequenced."""
    rng = random.Random(seed)
    rows = []
    seen = set()

    def spacer():
        while True:
            s = _rand_seq(rng, 20)
            if s not in seen and "GTTT" not in s:
                seen.add(s)
                return s

    scaffold_of = {}
    for t in TARGETS:
        for scaf in ("A", "C"):
            s = spacer()
            rows.append(
                {
                    "gene": t,
                    "seq": s,
                    "On-Target Efficacy Score": round(rng.uniform(0.2, 1.2), 3),
                    "BeginsWithG": s.startswith("G"),
                }
            )
            scaffold_of[s] = scaf
    for i in range(4):
        s = spacer()
        rows.append(
            {"gene": "NO-TARGET", "seq": s, "On-Target Efficacy Score": float("nan"), "BeginsWithG": s.startswith("G")}
        )
        scaffold_of[s] = "A" if i % 2 == 0 else "C"
    for t in ("GENEA", "NO-TARGET"):  # designed but never observed
        s = spacer()
        rows.append({"gene": t, "seq": s, "On-Target Efficacy Score": 0.1, "BeginsWithG": s.startswith("G")})
        scaffold_of[s] = "unsequenced"
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False)
    df["_scaffold_truth"] = df["seq"].map(scaffold_of)
    return df


def write_10x_h5(
    path: Path, counts: sp.csr_matrix, barcodes: List[str], gene_ids: List[str], gene_names: List[str]
) -> None:
    """Cell Ranger v3-style HDF5 (matrix stored features x barcodes, CSC)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = sp.csr_matrix(counts, dtype=np.int32)
    n_cells, n_genes = counts.shape
    with h5py.File(path, "w") as f:
        f.attrs["filetype"] = "matrix"
        f.attrs["version"] = 2
        f.attrs["chemistry_description"] = "synthetic"
        g = f.create_group("matrix")
        g.create_dataset("barcodes", data=np.array(barcodes, dtype="S"))
        g.create_dataset("data", data=counts.data.astype(np.int32))
        g.create_dataset("indices", data=counts.indices.astype(np.int64))
        g.create_dataset("indptr", data=counts.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array([n_genes, n_cells], dtype=np.int32))
        feat = g.create_group("features")
        feat.create_dataset("id", data=np.array(gene_ids, dtype="S"))
        feat.create_dataset("name", data=np.array(gene_names, dtype="S"))
        feat.create_dataset("feature_type", data=np.array(["Gene Expression"] * n_genes, dtype="S"))
        feat.create_dataset("genome", data=np.array(["GRCh38"] * n_genes, dtype="S"))
        feat.create_dataset("_all_tag_keys", data=np.array(["genome"], dtype="S"))


def gene_table() -> Tuple[List[str], List[str]]:
    names = [f"GENE{i:04d}" for i in range(N_GENES)]
    # duplicate symbol to exercise var_names_make_unique; MT / ribo genes for QC
    names[10] = "DUPGENE"
    names[11] = "DUPGENE"
    for i, mt in enumerate(["MT-CO1", "MT-CO2", "MT-ND1", "MT-ND2", "MT-CYB"]):
        names[N_GENES - 5 + i] = mt
    for i in range(6):
        names[200 + i] = f"RPS{i + 1}"
        names[210 + i] = f"RPL{i + 1}"
    ids = [f"ENSG{i:011d}" for i in range(N_GENES)]
    return ids, names


def make_well(
    outdir: Path,
    well_id: str,
    design: pd.DataFrame,
    n_cells: int = 320,
    n_doublets: int = 40,
    n_multiplet: int = 30,
    n_noguide: int = 15,
    seed: int = 0,
) -> Dict[str, object]:
    """Write one well's h5 + two guide FASTQ lanes; return paths and truth."""
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    ids, names = gene_table()
    n_single = n_cells - n_doublets
    # Two cell "types" with different expression programmes.
    base = rng.gamma(2.0, 1.0, size=N_GENES)
    prog = np.ones((2, N_GENES))
    prog[0, :80] *= 4.0
    prog[1, 80:160] *= 4.0
    prog[:, N_GENES - 5 :] *= 0.6
    types = rng.integers(0, 2, size=n_single)
    depth = rng.lognormal(np.log(6000), 0.35, size=n_single)
    singles = np.zeros((n_single, N_GENES), dtype=np.int32)
    for i in range(n_single):
        p = base * prog[types[i]]
        p /= p.sum()
        singles[i] = rng.multinomial(int(depth[i]), p)
    # doublets = heterotypic sums
    dbl = np.zeros((n_doublets, N_GENES), dtype=np.int32)
    t0 = np.where(types == 0)[0]
    t1 = np.where(types == 1)[0]
    for i in range(n_doublets):
        dbl[i] = singles[rng.choice(t0)] + singles[rng.choice(t1)]
    X = np.vstack([singles, dbl])
    # a few deliberately low-quality cells (low counts, high mt) among singlets
    low = rng.choice(n_single, 6, replace=False)
    X[low[:3]] = (X[low[:3]] * 0.05).astype(np.int32)
    X[low[3:], N_GENES - 5 :] += 3000
    barcodes = [f"{_rand_seq(prng, 16)}-1" for _ in range(n_cells)]
    assert len(set(barcodes)) == n_cells
    h5 = outdir / well_id / "filtered_feature_bc_matrix.h5"
    write_10x_h5(h5, sp.csr_matrix(X), barcodes, ids, names)
    # guide truth
    d = design[design["_scaffold_truth"] != "unsequenced"].reset_index(drop=True)
    A = d[d["_scaffold_truth"] == "A"]["seq"].tolist()
    C = d[d["_scaffold_truth"] == "C"]["seq"].tolist()
    spacer_scaf = dict(zip(d["seq"], d["_scaffold_truth"]))
    truth_umis: Dict[str, Dict[str, int]] = {}
    is_multiplet = np.zeros(n_cells, dtype=bool)
    no_guide = np.zeros(n_cells, dtype=bool)
    order = rng.permutation(n_cells)
    multiplet_cells = set(order[:n_multiplet].tolist())
    noguide_cells = set(order[n_multiplet : n_multiplet + n_noguide].tolist())
    for i, bc in enumerate(barcodes):
        bare = bc.split("-")[0]
        guides: Dict[str, int] = {}
        if i in noguide_cells:
            no_guide[i] = True
            # a single sub-threshold UMI on one guide for some of them
            if prng.random() < 0.5:
                guides[prng.choice(A)] = 1
        else:
            a = prng.choice(A)
            c = prng.choice(C)
            guides[a] = int(rng.integers(5, 40))
            guides[c] = int(rng.integers(5, 40))
            if i in multiplet_cells:
                is_multiplet[i] = True
                extra = prng.choice([g for g in A if g != a])
                guides[extra] = int(rng.integers(4, 30))
        truth_umis[bare] = guides
    # FASTQ reads
    reads: List[Tuple[str, str]] = []
    unknown_spacer = "ACGTACGTACGTACGTACGT"
    for bc, guides in truth_umis.items():
        for spacer, n_umi in guides.items():
            scaf = SCAFFOLDS[spacer_scaf[spacer]]
            for _ in range(n_umi):
                umi = _rand_seq(prng, 12)
                n_reads = 1 + int(rng.poisson(1.5))
                for _ in range(n_reads):  # PCR duplicates -> same UMI
                    gs = "G" * prng.choice([0, 1, 2, 2, 2, 3, 4])
                    reads.append((f"{bc}{umi}{TSO}{gs}{spacer}{scaf}"[:151], bc))
    # noise: invalid barcodes, unknown spacers, reads without an anchor
    for _ in range(300):
        bc = _rand_seq(prng, 16)
        reads.append((f"{bc}{_rand_seq(prng, 12)}{TSO}GG{prng.choice(A)}{SCAFFOLDS['A']}"[:151], "invalid"))
    for _ in range(200):
        bc = barcodes[prng.randrange(n_cells)].split("-")[0]
        reads.append((f"{bc}{_rand_seq(prng, 12)}{TSO}GG{unknown_spacer}{SCAFFOLDS['C']}"[:151], "unknown"))
    for _ in range(100):
        reads.append((_rand_seq(prng, 151), "noanchor"))
    prng.shuffle(reads)
    fq_dir = outdir / f"{well_id}F" / "fastq"
    fq_dir.mkdir(parents=True, exist_ok=True)
    half = len(reads) // 2
    files = []
    for lane, chunk in (("L001", reads[:half]), ("L002", reads[half:])):
        p = fq_dir / f"{well_id}F_S1_{lane}_R1_001.fastq.gz"
        with gzip.open(p, "wt") as fh:
            for k, (seq, _) in enumerate(chunk):
                fh.write(f"@synthetic:{well_id}:{lane}:{k} 1:N:0:ACGT\n{seq}\n+\n{'F' * len(seq)}\n")
        files.append(p)
    return {
        "well_id": well_id,
        "h5": h5,
        "fastq_dir": fq_dir,
        "fastqs": files,
        "barcodes": barcodes,
        "n_cells": n_cells,
        "truth_umis": truth_umis,
        "is_doublet": np.r_[np.zeros(n_single, bool), np.ones(n_doublets, bool)],
        "is_guide_multiplet": is_multiplet,
        "no_guide": no_guide,
        "n_reads": len(reads),
    }


def make_dataset(outdir: Path, wells: Tuple[str, ...] = ("W1", "W2"), n_cells: int = 320) -> Dict[str, object]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    design_path = outdir / "design.xlsx"
    design = make_design(design_path, seed=1)
    out = {"design_path": design_path, "design": design, "wells": {}}
    for i, w in enumerate(wells):
        out["wells"][w] = make_well(outdir, w, design, n_cells=n_cells, seed=10 + i)
    return out


def basic_qc_config(dataset: Dict[str, object], outdir: Path, **overrides) -> Dict[str, object]:
    """Config dict for :func:`perturbseq_pipeline.config.Config.from_dict`."""
    samples = {}
    for i, (w, info) in enumerate(dataset["wells"].items()):
        samples[w] = {
            "gex_h5": str(info["h5"]),
            "guide_library": f"{w}F",
            "guide_fastq_dir": str(info["fastq_dir"]),
            "condition_code": "COND1" if i == 0 else "COND2",
            "gem_well": "A" if i == 0 else "B",
        }
    cfg = {
        "run": {"name": "synthetic_qc", "outdir": str(outdir), "seed": 0, "stop_after": "qc"},
        "samples": samples,
        "qc": {
            "thresholds": {
                "method": "mad",
                "n_mads": 3,
                "min_genes_floor": 50,
                "min_counts_floor": 200,
                "max_pct_mt": 20.0,
            },
            "doublets": {"enabled": True, "threshold": 0.15},
        },
        "guides": {
            "design": {"path": str(dataset["design_path"])},
            "fastq": {"read_pattern": "*_R1_*.fastq.gz", "n_workers": 1},
            "detection_threshold": 3,
        },
        "output": {"h5ad_name": "synthetic_qc.h5ad"},
        "report": {"figure_dpi": 60},
    }
    for key, value in overrides.items():
        cfg.setdefault(key, {}).update(value)
    return cfg
