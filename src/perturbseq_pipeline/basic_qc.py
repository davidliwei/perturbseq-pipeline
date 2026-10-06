"""Basic QC stage: load -> guide quantification -> per-sample QC -> flags -> concat -> write.

This stage measures and annotates quality; it does not clean aggressively.

    10x GEX (per GEM well)
        -> optional guide quantification (FASTQ | matrix | none)
        -> per-sample AnnData loading + metadata
        -> per-sample expression QC metrics and flags        (gex_qc_pass)
        -> per-sample doublet detection                      (flag only)
        -> guide QC / guide-derived multiplet detection      (flag only)
        -> per-sample all-cell checkpoints
        -> concatenation (not integration)
        -> combined all-cells object + expression-QC object
        -> tables, figures, HTML report, provenance
        -> STOP

Invariants enforced here and checked by the tests:

* every Cell Ranger-called cell of every sample is present in the per-sample
  checkpoints and in the combined all-cells object;
* ``predicted_doublet`` and ``guide_multiplet_flag`` never influence which
  cells are written — the expression-QC object subsets on ``gex_qc_pass``
  only, so predicted doublets and guide multiplets remain in it;
* ``X`` and ``layers['counts']`` stay raw integer counts; nothing is
  normalised, embedded, clustered or integrated;
* guides are identified and counted, but no perturbation is assigned.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import provenance as prov_mod
from .config import Config, SampleConfig
from .doublets import DOUBLET_SCORE, PREDICTED_DOUBLET, run_scrublet
from .guide_counting import (
    GuideCountJob,
    GuideCountResult,
    count_guides,
    find_guide_fastqs,
    guide_counts_from_matrix,
    write_guide_counts,
)
from .guide_design import load_guide_design
from .guide_qc import (
    GUIDE_FLAG_COLUMNS,
    attach_guide_counts,
    empty_guide_annotations,
    guide_detection_sensitivity,
    guide_sample_summary,
    infer_scaffold_classes,
    scaffold_classes,
    scrublet_vs_guide_multiplet_table,
)
from .io import BARCODE_KEY, LANE_KEY, read_10x_h5, read_10x_mtx_sample, strip_barcode_suffix, write_h5ad
from .qc import (
    GEX_QC_PASS,
    QC_FLAG_COLUMNS,
    compute_basic_qc_metrics,
    expression_qc_step_table,
    flag_expression_qc,
    resolve_sample_thresholds,
)

logger = logging.getLogger(__name__)

STAGE_NAME = "basic_qc"

SUBDIRS = ("guide_counts", "per_sample", "combined", "figures", "tables", "reports", "logs")

#: Sample-level obs columns written for every cell.
SAMPLE_OBS_COLUMNS = ("sample_id", "condition_code", "gem_well", "guide_library", BARCODE_KEY)


@dataclass
class BasicQCResult:
    outdir: Path
    allcells_h5ad: Path
    pass_h5ad: Path
    report: Path
    per_sample_h5ads: Dict[str, Path] = field(default_factory=dict)
    guide_count_dirs: Dict[str, Path] = field(default_factory=dict)
    tables: Dict[str, Path] = field(default_factory=dict)
    provenance_json: Optional[Path] = None
    n_cells_all: int = 0
    n_cells_pass: int = 0
    n_genes: int = 0
    n_predicted_doublets_all: int = 0
    n_predicted_doublets_pass: int = 0
    n_guide_multiplets_all: int = 0
    n_guide_multiplets_pass: int = 0
    sample_summary: Optional[pd.DataFrame] = None
    guide_summary: Optional[pd.DataFrame] = None
    runtime_seconds: float = 0.0
    #: Kept for in-process callers/tests; ``None`` when not retained.
    adata_all: Optional[ad.AnnData] = None
    adata_pass: Optional[ad.AnnData] = None


# Sample loading


def _load_sample(sample_id: str, smp: SampleConfig, cfg: Config) -> ad.AnnData:
    if smp.gex_h5:
        adata = read_10x_h5(
            smp.gex_h5,
            sample_id,
            var_names=cfg.input.var_names,
            gex_feature_type=cfg.input.gex_feature_type,
            feature_type_column=cfg.input.feature_type_column,
        )
    else:
        adata = read_10x_mtx_sample(
            smp.gex_mtx_dir,
            sample_id,
            var_names=cfg.input.var_names,
            gex_feature_type=cfg.input.gex_feature_type,
            feature_type_column=cfg.input.feature_type_column,
        )
    return attach_sample_annotations(adata, sample_id, smp)


def attach_sample_annotations(adata: ad.AnnData, sample_id: str, smp: SampleConfig) -> ad.AnnData:
    """Constant per-sample metadata as categorical obs columns (neutral labels)."""
    n = adata.n_obs
    adata.obs["sample_id"] = pd.Categorical([sample_id] * n)
    adata.obs[LANE_KEY] = pd.Categorical([sample_id] * n)
    adata.obs["condition_code"] = pd.Categorical(
        [str(smp.condition_code) if smp.condition_code is not None else "NA"] * n
    )
    adata.obs["gem_well"] = pd.Categorical([str(smp.gem_well) if smp.gem_well is not None else "NA"] * n)
    adata.obs["guide_library"] = pd.Categorical(
        [str(smp.guide_library) if smp.guide_library is not None else "none"] * n
    )
    for key, val in (smp.metadata or {}).items():
        if key in adata.obs.columns:
            logger.warning(
                "%s: metadata key %r collides with an existing obs column; stored as %r", sample_id, key, f"{key}_meta"
            )
            key = f"{key}_meta"
        adata.obs[key] = pd.Categorical([str(val)] * n)
    return adata


def _legacy_samples(
    cfg: Config,
) -> Tuple[Dict[str, SampleConfig], Dict[str, ad.AnnData], Dict[str, Optional[ad.AnnData]]]:
    """Adapter: run the basic QC stage on legacy ``input.mtx_dirs`` / ``input.h5ad`` data.

    The loaded object is split by lane; guide features already present in the
    input become per-sample guide matrices (source ``matrix``).
    """
    from . import io as io_mod

    data = io_mod.load_data(cfg)
    expr, guides = data.expr, data.guides
    lanes = expr.obs[LANE_KEY].astype(str)
    samples: Dict[str, SampleConfig] = {}
    adatas: Dict[str, ad.AnnData] = {}
    guide_adatas: Dict[str, Optional[ad.AnnData]] = {}
    for lane in pd.unique(lanes):
        mask = (lanes == lane).to_numpy()
        sub = expr[mask].copy()
        if BARCODE_KEY not in sub.obs:
            sub.obs[BARCODE_KEY] = sub.obs_names.astype(str).to_numpy()
        if "counts" in sub.layers:
            sub.X = sub.layers["counts"].copy()
        sub.X = sp.csr_matrix(sub.X)
        sub.layers["counts"] = sub.X.copy()
        cond = str(sub.obs["condition"].iloc[0]) if "condition" in sub.obs else None
        smp = SampleConfig(
            gex_mtx_dir=str(data.lanes.get(lane, "")), condition_code=cond, gem_well=None, guide_library=None
        )
        adatas[lane] = attach_sample_annotations(sub, lane, smp)
        samples[lane] = smp
        guide_adatas[lane] = guides[mask].copy() if guides is not None else None
    return samples, adatas, guide_adatas


def _guide_result_from_anndata(sample_id: str, gad: ad.AnnData, cfg: Config) -> Tuple[GuideCountResult, pd.DataFrame]:
    """Wrap an in-memory guide AnnData (legacy split features) as a count result."""
    from .guide_design import CONTROL_TARGET_LABEL, is_control_label

    ids = gad.var_names.astype(str).to_numpy()
    design = pd.DataFrame(
        {
            "guide_id": ids,
            "protospacer": "",
            "target_raw": gad.var["guide_symbol"].astype(str).to_numpy() if "guide_symbol" in gad.var else ids,
            "scaffold": "unknown",
            "scaffold_source": "unspecified",
            "design_index": np.arange(len(ids)),
        }
    )
    design["is_control"] = is_control_label(design["target_raw"].tolist(), cfg.guides.ntc_patterns)
    design["target"] = np.where(design["is_control"], CONTROL_TARGET_LABEL, design["target_raw"])
    counts = sp.csr_matrix(gad.X, dtype=np.int32)
    res = GuideCountResult(
        sample_id=sample_id,
        guide_ids=list(ids),
        cell_barcodes=list(gad.obs_names.astype(str)),
        counts=counts,
        guide_reads=np.zeros(len(ids), dtype=np.int64),
        guide_scaffold_reads=np.zeros((len(ids), len(cfg.guides.fastq.scaffolds)), dtype=np.int64),
        scaffold_names=[str(k) for k in cfg.guides.fastq.scaffolds],
        stats={
            "source": "input matrix",
            "guides_designed": len(ids),
            "guides_detected_any_umi": int((counts.sum(axis=0) > 0).sum()),
            "total_guide_umis_in_matrix": int(counts.sum()),
            "cells_in_gex_universe": gad.n_obs,
        },
        source="matrix",
    )
    return res, design


def _guide_source_for(smp: SampleConfig, cfg: Config) -> str:
    src = cfg.guides.source
    if src == "none":
        return "none"
    if src == "fastq":
        if not (smp.guide_fastq_dir or smp.guide_fastqs):
            raise ValueError("guides.source is 'fastq' but the sample has no guide_fastq_dir/guide_fastqs")
        return "fastq"
    if src == "matrix":
        if not smp.guide_matrix:
            raise ValueError("guides.source is 'matrix' but the sample has no guide_matrix")
        return "matrix"
    if smp.guide_fastq_dir or smp.guide_fastqs:
        return "fastq"
    if smp.guide_matrix:
        return "matrix"
    return "none"


# Stage driver


def run_basic_qc(
    cfg: Config, outdir: Path, registry, config_path: Optional[str] = None, keep_adata: bool = False
) -> BasicQCResult:
    """Run the basic QC stage. See the module docstring for the contract."""
    from . import qc_plots

    t0 = time.time()
    outdir = Path(outdir)
    for sub in SUBDIRS:
        (outdir / sub).mkdir(parents=True, exist_ok=True)
    tabledir = outdir / "tables"
    table_paths: Dict[str, Path] = {}
    warnings: List[str] = []

    def _write_table(name: str, df: Optional[pd.DataFrame]) -> Optional[Path]:
        if df is None or len(df) == 0:
            return None
        p = tabledir / f"{name}.tsv"
        df.to_csv(p, sep="\t", index=False)
        table_paths[name] = p
        return p

    # 1. Samples and GEX loading
    samples = cfg.resolved_samples()
    legacy_guides: Dict[str, Optional[ad.AnnData]] = {}
    adatas: Dict[str, ad.AnnData] = {}
    if samples:
        logger.info("Basic QC: %d sample(s) from config: %s", len(samples), list(samples))
        for sid, smp in samples.items():
            adatas[sid] = _load_sample(sid, smp, cfg)
    else:
        logger.info("Basic QC: no 'samples' block; splitting legacy input by lane")
        samples, adatas, legacy_guides = _legacy_samples(cfg)
    input_cells = {sid: int(a.n_obs) for sid, a in adatas.items()}
    # 2. Guide design + quantification
    design: Optional[pd.DataFrame] = None
    if cfg.guides.design.path:
        design = load_guide_design(cfg)
    guide_sources = {sid: _guide_source_for(smp, cfg) for sid, smp in samples.items()}
    if legacy_guides:
        for sid, g in legacy_guides.items():
            guide_sources[sid] = "matrix" if g is not None else "none"
    guide_results: Dict[str, GuideCountResult] = {}
    guide_inputs: Dict[str, object] = {}
    fastq_jobs: List[GuideCountJob] = []
    for sid, smp in samples.items():
        if guide_sources[sid] != "fastq":
            continue
        if design is None:
            raise ValueError("Guide FASTQ counting requires guides.design.path")
        files = (
            list(smp.guide_fastqs)
            if smp.guide_fastqs
            else find_guide_fastqs(smp.guide_fastq_dir, smp.guide_fastq_pattern or cfg.guides.fastq.read_pattern)
        )
        guide_inputs[sid] = files
        bare = strip_barcode_suffix(adatas[sid].obs[BARCODE_KEY].astype(str), cfg.guides.fastq.barcode_suffix_regex)
        if pd.Index(bare).has_duplicates:
            raise ValueError(
                f"{sid}: stripping the barcode suffix produced duplicate barcodes; check guides.fastq.barcode_suffix_regex"
            )
        fastq_jobs.append(GuideCountJob(sample_id=sid, fastq_files=files, cell_barcodes=list(bare)))
    if fastq_jobs:
        guide_results.update(count_guides(fastq_jobs, design, cfg))
        # scaffold-specific features: the shared reference becomes the feature-level design
        fd = next((r.feature_design for r in guide_results.values() if r.feature_design is not None), None)
        if fd is not None:
            design = fd
    for sid, smp in samples.items():
        if guide_sources[sid] != "matrix":
            continue
        bare = strip_barcode_suffix(adatas[sid].obs[BARCODE_KEY].astype(str), cfg.guides.fastq.barcode_suffix_regex)
        if sid in legacy_guides and legacy_guides[sid] is not None:
            res, d = _guide_result_from_anndata(sid, legacy_guides[sid], cfg)
            res.cell_barcodes = list(adatas[sid].obs[BARCODE_KEY].astype(str))
        else:
            guide_inputs[sid] = smp.guide_matrix
            res, d = guide_counts_from_matrix(sid, smp.guide_matrix, list(bare), design, cfg)
        if design is None:
            design = d
        elif list(d["guide_id"]) != list(design["guide_id"]):
            raise ValueError(f"{sid}: guide matrix features differ from the shared guide reference")
        guide_results[sid] = res
    if design is not None:
        design = infer_scaffold_classes(design, guide_results, cfg)
    classes = scaffold_classes(design) if design is not None else []
    guide_count_dirs: Dict[str, Path] = {}
    for sid, res in guide_results.items():
        gdir = outdir / "guide_counts" / sid
        write_guide_counts(res, design, gdir)
        guide_count_dirs[sid] = gdir
    # 3. Per-sample QC (metrics, flags, doublets, guides) + checkpoints
    thresholds_all: Dict[str, Dict[str, object]] = {}
    doublet_rows: List[Dict[str, object]] = []
    guide_rows: List[Dict[str, object]] = []
    flag_tables: List[pd.DataFrame] = []
    sample_rows: List[Dict[str, object]] = []
    per_sample_h5ads: Dict[str, Path] = {}
    umis_by_sample: Dict[str, np.ndarray] = {}
    sensitivity_tables: List[pd.DataFrame] = []
    for sid, expr in adatas.items():
        smp = samples[sid]
        n_in = expr.n_obs
        logger.info("=== Basic QC: sample %s (%d cells) ===", sid, n_in)
        compute_basic_qc_metrics(expr, cfg)
        cond = (
            expr.obs[cfg.qc.thresholds.condition_key].iloc[0] if cfg.qc.thresholds.condition_key in expr.obs else None
        )
        thr = resolve_sample_thresholds(expr.obs, cfg, sid, cond)
        thresholds_all[sid] = thr
        flag_expression_qc(expr, thr)
        flag_tables.append(expression_qc_step_table(expr, sid))
        dsum = run_scrublet(expr, cfg, sid, seed=cfg.run.seed)
        doublet_rows.append(dsum)
        if dsum.get("threshold_failed"):
            warnings.append(
                f"{sid}: Scrublet automatic threshold failed; predicted_doublet is False for all cells (scores retained)."
            )
        elif dsum.get("threshold_suspect"):
            warnings.append(
                f"{sid}: Scrublet automatic threshold {dsum.get('threshold'):.3f} looks implausible "
                f"({dsum.get('threshold_suspect_reason')}); {dsum.get('n_predicted_doublets')} cells called. "
                "Scores are stored for every cell; choose a threshold after inspection (qc.doublets.threshold)."
            )
        res = guide_results.get(sid)
        if res is not None:
            attach_guide_counts(expr, res, design, cfg)
            umis_by_sample[sid] = res.guide_umis()
            sensitivity_tables.append(
                guide_detection_sensitivity(expr.obsm[cfg.output.guide_obsm_key], design, cfg, sid)
            )
        else:
            empty_guide_annotations(expr, cfg, classes)
            if design is not None:
                expr.obsm[cfg.output.guide_obsm_key] = sp.csr_matrix((expr.n_obs, len(design)), dtype=np.int32)
                expr.uns["guide_names"] = list(design["guide_id"].astype(str))
        guide_rows.append(guide_sample_summary(expr, sid, res, design, cfg))
        expr.uns["sample"] = prov_mod.uns_safe(
            {
                "sample_id": sid,
                "gex_input": smp.gex_path() if (smp.gex_h5 or smp.gex_mtx_dir) else "legacy",
                "guide_library": smp.guide_library,
                "guide_source": guide_sources[sid],
                "guide_input": guide_inputs.get(sid),
                "condition_code": smp.condition_code,
                "gem_well": smp.gem_well,
            }
        )
        expr.uns["qc_thresholds"] = prov_mod.uns_safe(thr)
        expr.uns["basic_qc"] = {
            "stage": STAGE_NAME,
            "cells_removed": 0,
            "doublets_removed": 0,
            "guide_multiplets_removed": 0,
        }
        if design is not None:
            expr.uns["guide_features"] = _uns_frame(design)
        assert expr.n_obs == n_in, f"{sid}: cell count changed during basic QC ({n_in} -> {expr.n_obs})"
        p = outdir / "per_sample" / f"{sid}_qc_allcells.h5ad"
        write_h5ad(expr, p)
        per_sample_h5ads[sid] = p
        obs = expr.obs
        qc_plots.plot_sample_expression_qc(obs, thr, sid, registry)
        qc_plots.plot_doublet_scores(obs, sid, dsum.get("threshold"), registry)
        if res is not None:
            qc_plots.plot_guide_qc(obs, sid, classes, registry)
            qc_plots.plot_scrublet_vs_guide(obs, sid, registry)
        sample_rows.append(_sample_summary_row(expr, sid, smp, thr, dsum, guide_sources[sid]))
    # 4. Concatenate (not integrate)
    sample_ids = list(adatas)
    logger.info("Concatenating %d sample(s): %s", len(sample_ids), sample_ids)
    var_ref = adatas[sample_ids[0]].var
    for sid in sample_ids[1:]:
        if not adatas[sid].var_names.equals(var_ref.index):
            raise ValueError(f"{sid}: gene set differs from {sample_ids[0]}; all samples must share one reference")
    combined = ad.concat(
        [adatas[s] for s in sample_ids],
        axis=0,
        join="inner",
        merge="same",
        uns_merge=None,
        label=None,
        index_unique=None,
    )
    for col in ("gene_ids", "feature_types", "genome", "mt", "ribo", "hb", "gene_symbols"):
        if col in var_ref.columns and col not in combined.var.columns:
            combined.var[col] = var_ref[col].to_numpy()
    combined.var["n_cells_by_counts"] = np.asarray(combined.X.getnnz(axis=0)).ravel()
    for s in sample_ids:
        combined.var[f"n_cells_{s}"] = np.asarray(adatas[s].X.getnnz(axis=0)).ravel()
    if combined.obs_names.has_duplicates:
        raise ValueError("Combined object has duplicate cell identifiers")
    combined.obs["sample_id"] = pd.Categorical(combined.obs["sample_id"].astype(str), categories=sample_ids)
    combined.obs[LANE_KEY] = combined.obs["sample_id"]
    if cfg.output.guide_obsm_key in combined.obsm:
        combined.obsm[cfg.output.guide_obsm_key] = sp.csr_matrix(
            combined.obsm[cfg.output.guide_obsm_key], dtype=np.int32
        )
    n_all = combined.n_obs
    assert n_all == sum(input_cells.values()), "combined all-cells object lost cells"
    del adatas
    provenance = prov_mod.collect(
        config_path=config_path,
        inputs={
            "samples": {
                sid: {
                    "gex": smp.gex_path() if (smp.gex_h5 or smp.gex_mtx_dir) else "legacy",
                    "guide_library": smp.guide_library,
                    "guide_source": guide_sources[sid],
                    "guide_input": guide_inputs.get(sid),
                    "input_cells": input_cells[sid],
                }
                for sid, smp in samples.items()
            },
            "guide_design": cfg.guides.design.path,
        },
        extra={
            "stage": STAGE_NAME,
            "pipeline_version": _pipeline_version(),
            "run_name": cfg.run.name,
            "outdir": str(outdir),
        },
    )
    combined.uns["provenance"] = prov_mod.uns_safe(provenance)
    combined.uns["config"] = _config_yaml(cfg)
    combined.uns["sample_manifest"] = prov_mod.uns_safe(
        {
            sid: {
                "gex": smp.gex_path() if (smp.gex_h5 or smp.gex_mtx_dir) else "legacy",
                "guide_library": smp.guide_library,
                "guide_source": guide_sources[sid],
                "condition_code": smp.condition_code,
                "gem_well": smp.gem_well,
                "input_cells": input_cells[sid],
            }
            for sid, smp in samples.items()
        }
    )
    combined.uns["qc_thresholds"] = prov_mod.uns_safe(thresholds_all)
    combined.uns["guide_sources"] = prov_mod.uns_safe(dict(guide_sources))
    combined.uns["guide_source"] = (
        next(iter(set(guide_sources.values()))) if len(set(guide_sources.values())) == 1 else "mixed"
    )
    combined.uns["scrublet"] = prov_mod.uns_safe({r["sample_id"]: r for r in doublet_rows})
    combined.uns["basic_qc"] = prov_mod.uns_safe(
        {
            "stage": STAGE_NAME,
            "cells_removed": 0,
            "doublets_removed": 0,
            "guide_multiplets_removed": 0,
            "gex_qc_pass_rule": "NOT(" + " OR ".join(QC_FLAG_COLUMNS) + ")",
            "note": "predicted_doublet and guide_multiplet_flag are annotations only",
        }
    )
    if design is not None:
        combined.uns["guide_features"] = _uns_frame(design)
        combined.uns["guide_names"] = list(design["guide_id"].astype(str))
        combined.uns["guide_target_genes"] = list(design["target"].astype(str))
        combined.uns["guide_qc"] = prov_mod.uns_safe(
            {
                "detection_threshold_umi": int(
                    cfg.guides.multiplet.detection_threshold or cfg.guides.detection_threshold
                ),
                "scaffold_classes": classes,
                "perturbation_assignment": "not performed in basic QC",
            }
        )
    # 5. Write combined objects
    stem = Path(cfg.output.h5ad_name).stem
    pass_name = cfg.output.h5ad_name
    all_name = cfg.output.unfiltered_h5ad_name or f"{stem}_allcells.h5ad"
    all_path = outdir / "combined" / all_name
    pass_path = outdir / "combined" / pass_name
    write_h5ad(combined, all_path)
    pass_mask = combined.obs[GEX_QC_PASS].to_numpy(dtype=bool)
    passed = combined[pass_mask].copy()
    passed.uns["basic_qc"] = prov_mod.uns_safe(
        {
            **{k: v for k, v in combined.uns["basic_qc"].items()},
            "subset": f"{GEX_QC_PASS} == True (expression criteria only; doublets and guide multiplets retained)",
            "cells_removed": int(n_all - passed.n_obs),
        }
    )
    write_h5ad(passed, pass_path)
    n_pass = passed.n_obs
    n_dbl_all = int(combined.obs[PREDICTED_DOUBLET].astype(bool).sum())
    n_dbl_pass = int(passed.obs[PREDICTED_DOUBLET].astype(bool).sum())
    n_gm_all = int(combined.obs["guide_multiplet_flag"].astype(bool).sum())
    n_gm_pass = int(passed.obs["guide_multiplet_flag"].astype(bool).sum())
    logger.info(
        "Combined: %d cells (all) / %d cells (gex_qc_pass); predicted doublets retained %d / %d; "
        "guide multiplets retained %d / %d",
        n_all,
        n_pass,
        n_dbl_all,
        n_dbl_pass,
        n_gm_all,
        n_gm_pass,
    )
    # 6. Tables
    sample_summary = pd.DataFrame(sample_rows)
    _write_table("sample_qc_summary", sample_summary)
    cell_cols = [c for c in combined.obs.columns]
    cell_table = combined.obs[cell_cols].copy()
    cell_table.insert(0, "cell_id", combined.obs_names.to_numpy())
    _write_table("cell_qc_summary", cell_table)
    guide_summary = pd.DataFrame(guide_rows)
    _write_table("guide_qc_summary", guide_summary)
    filtering = _filtering_summary(combined.obs, passed.obs, sample_ids)
    _write_table("filtering_summary", filtering)
    doublet_summary = pd.DataFrame(doublet_rows)
    _write_table("doublet_summary", doublet_summary)
    thr_table = pd.DataFrame(list(thresholds_all.values()))
    _write_table("qc_thresholds", thr_table)
    svg = scrublet_vs_guide_multiplet_table(combined.obs, by="sample_id")
    svg_all = scrublet_vs_guide_multiplet_table(combined.obs)
    svg_table = pd.concat([svg, svg_all], ignore_index=True) if len(svg) else svg_all
    _write_table("scrublet_vs_guide_multiplet", svg_table)
    _write_table("expression_qc_flags", pd.concat(flag_tables, ignore_index=True) if flag_tables else None)
    if design is not None:
        feat = design.copy()
        for sid, res in guide_results.items():
            feat[f"umis_{sid}"] = res.guide_umis()
            feat[f"cells_positive_{sid}"] = res.guide_positive_cells(
                int(cfg.guides.multiplet.detection_threshold or cfg.guides.detection_threshold)
            )
        _write_table("guide_feature_table", feat)
    sensitivity = pd.concat(sensitivity_tables, ignore_index=True) if sensitivity_tables else pd.DataFrame()
    _write_table("guide_detection_sensitivity", sensitivity)
    per_file = [dict(sample_id=sid, **st) for sid, res in guide_results.items() for st in res.per_file]
    _write_table("guide_counting_per_file", pd.DataFrame(per_file) if per_file else None)
    # 7. Combined figures, provenance, report
    qc_plots.plot_combined_metrics(combined.obs, registry)
    qc_plots.plot_sample_flag_summary(sample_summary, registry)
    if guide_results:
        qc_plots.plot_scrublet_vs_guide(combined.obs, "combined", registry)
        qc_plots.plot_guide_representation(design, umis_by_sample, registry)
        qc_plots.plot_guide_detection_sensitivity(sensitivity, registry)
    qc_plots.plot_doublet_score_by_sample(combined.obs, doublet_rows, registry)
    registry.manifest().to_csv(tabledir / "figure_manifest.csv", index=False)
    prov_path = prov_mod.write_json(provenance, outdir / "reports" / "provenance.json")
    from .report import build_qc_report

    outputs = {
        "all-cells h5ad": str(all_path),
        "expression-QC h5ad": str(pass_path),
        **{f"per-sample h5ad ({sid})": str(p) for sid, p in per_sample_h5ads.items()},
        **{f"guide counts ({sid})": str(p) for sid, p in guide_count_dirs.items()},
        "tables": str(tabledir),
        "figures": str(registry.figdir),
        "provenance": str(prov_path),
    }
    cards = [
        ("Samples", f"{len(sample_ids)}"),
        ("Input cells", f"{n_all:,}"),
        ("Genes", f"{combined.n_vars:,}"),
        ("Expression-QC pass", f"{n_pass:,} ({100 * n_pass / max(n_all, 1):.1f}%)"),
        ("Predicted doublets (flagged, retained)", f"{n_dbl_all:,} all / {n_dbl_pass:,} in QC object"),
        ("Guide multiplets (flagged, retained)", f"{n_gm_all:,} all / {n_gm_pass:,} in QC object"),
        ("Designed guides", f"{len(design):,}" if design is not None else "n/a"),
        ("Guide-detected cells", f"{int(combined.obs['guide_detected'].sum()):,}"),
    ]
    report_tables = {
        "sample_qc_summary": sample_summary,
        "qc_thresholds": thr_table,
        "filtering_summary": filtering,
        "doublet_summary": doublet_summary,
        "scrublet_vs_guide_multiplet": svg_table,
        "guide_qc_summary": guide_summary,
        "guide_detection_sensitivity": sensitivity,
    }
    report_path = build_qc_report(
        cfg,
        registry,
        outdir / "reports" / cfg.output.report_name,
        tables=report_tables,
        cards=cards,
        outputs=outputs,
        warnings=warnings,
        provenance_text=json.dumps(provenance, indent=2, default=str),
    )
    result = BasicQCResult(
        outdir=outdir,
        allcells_h5ad=all_path,
        pass_h5ad=pass_path,
        report=report_path,
        per_sample_h5ads=per_sample_h5ads,
        guide_count_dirs=guide_count_dirs,
        tables=table_paths,
        provenance_json=prov_path,
        n_cells_all=n_all,
        n_cells_pass=n_pass,
        n_genes=combined.n_vars,
        n_predicted_doublets_all=n_dbl_all,
        n_predicted_doublets_pass=n_dbl_pass,
        n_guide_multiplets_all=n_gm_all,
        n_guide_multiplets_pass=n_gm_pass,
        sample_summary=sample_summary,
        guide_summary=guide_summary,
        runtime_seconds=time.time() - t0,
        adata_all=combined if keep_adata else None,
        adata_pass=passed if keep_adata else None,
    )
    logger.info("Basic QC stage finished in %.0f s; stopping (run.stop_after = qc)", result.runtime_seconds)
    return result


# Helpers


def _pipeline_version() -> str:
    from . import __version__

    return __version__


def _config_yaml(cfg: Config) -> str:
    import yaml

    return yaml.safe_dump(cfg.to_dict(), sort_keys=False, default_flow_style=False)


def _uns_frame(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of ``df`` with h5ad-writable dtypes (strings, numbers, bools)."""
    out = df.copy().reset_index(drop=True)
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str)
        elif str(out[c].dtype) == "boolean":
            out[c] = out[c].fillna(False).astype(bool)
    out.index = out.index.astype(str)
    return out


def _sample_summary_row(
    expr: ad.AnnData, sid: str, smp: SampleConfig, thr: Dict[str, object], dsum: Dict[str, object], guide_source: str
) -> Dict[str, object]:
    obs = expr.obs
    n = expr.n_obs
    row: Dict[str, object] = {
        "sample_id": sid,
        "condition_code": smp.condition_code,
        "gem_well": smp.gem_well,
        "guide_library": smp.guide_library,
        "guide_source": guide_source,
        "input_cells": n,
        "n_genes": expr.n_vars,
        "median_total_counts": float(obs["total_counts"].median()),
        "median_n_genes": float(obs["n_genes_by_counts"].median()),
        "median_pct_mt": float(obs["pct_counts_mt"].median()) if "pct_counts_mt" in obs else float("nan"),
        "median_pct_ribo": float(obs["pct_counts_ribo"].median()) if "pct_counts_ribo" in obs else float("nan"),
        "min_counts": thr.get("min_counts"),
        "max_counts": thr.get("max_counts"),
        "min_genes": thr.get("min_genes"),
        "max_genes": thr.get("max_genes"),
        "max_pct_mt": thr.get("max_pct_mt"),
    }
    for col in QC_FLAG_COLUMNS:
        row[col] = int(obs[col].sum())
    row["gex_qc_pass"] = int(obs[GEX_QC_PASS].sum())
    row["frac_gex_qc_pass"] = row["gex_qc_pass"] / max(n, 1)
    row["predicted_doublet"] = int(obs[PREDICTED_DOUBLET].astype(bool).sum())
    row["frac_predicted_doublet"] = row["predicted_doublet"] / max(n, 1)
    row["scrublet_threshold"] = dsum.get("threshold")
    row["predicted_doublet_in_gex_pass"] = int((obs[PREDICTED_DOUBLET].astype(bool) & obs[GEX_QC_PASS]).sum())
    for col in GUIDE_FLAG_COLUMNS:
        row[col] = int(obs[col].astype(bool).sum())
    row["frac_guide_detected"] = row["guide_detected"] / max(n, 1)
    row["frac_guide_multiplet_flag"] = row["guide_multiplet_flag"] / max(n, 1)
    row["guide_multiplet_in_gex_pass"] = int((obs["guide_multiplet_flag"].astype(bool) & obs[GEX_QC_PASS]).sum())
    row["cells_removed_in_this_stage"] = 0
    return row


def _filtering_summary(obs_all: pd.DataFrame, obs_pass: pd.DataFrame, sample_ids: List[str]) -> pd.DataFrame:
    rows = []
    groups = [
        (s, obs_all[obs_all["sample_id"].astype(str) == s], obs_pass[obs_pass["sample_id"].astype(str) == s])
        for s in sample_ids
    ]
    groups.append(("ALL", obs_all, obs_pass))
    for name, a, p in groups:
        row = {
            "sample_id": name,
            "input_cells": len(a),
            "gex_qc_pass": len(p),
            "removed_by_expression_qc": len(a) - len(p),
            "frac_retained": len(p) / max(len(a), 1),
        }
        for col in QC_FLAG_COLUMNS:
            row[f"flag_{col}"] = int(a[col].sum())
        row["predicted_doublets_allcells"] = int(a[PREDICTED_DOUBLET].astype(bool).sum())
        row["predicted_doublets_in_qc_object"] = int(p[PREDICTED_DOUBLET].astype(bool).sum())
        row["guide_multiplets_allcells"] = int(a["guide_multiplet_flag"].astype(bool).sum())
        row["guide_multiplets_in_qc_object"] = int(p["guide_multiplet_flag"].astype(bool).sum())
        row["doublets_removed"] = 0
        row["guide_multiplets_removed"] = 0
        rows.append(row)
    return pd.DataFrame(rows)
