"""Master perturbation meta-analysis table.

Combines orthogonal target-level biological dimensions into a single
master reference table: ``tables/perturbation_meta.csv``.

The biological dimensions merged include:
1. Perturbation efficacy: target gene knockdown strength and FDR
   (from :mod:`perturbseq_pipeline.perturbation`)
2. Penetrance: single-cell perturbation response scores and responder fractions
   (from :mod:`perturbseq_pipeline.ps_score`)
3. Topology: continuous manifold neighbourhood enrichment
   (from :mod:`perturbseq_pipeline.lochness`)
4. Phenotype magnitude: Energy Distance and permutation DistanceTest significance
   (from :mod:`perturbseq_pipeline.distance`)
5. Mechanistic organization: Co-functional/gene-effect modules and Phenotype-distance modules
   (from :mod:`perturbseq_pipeline.modules` and :mod:`perturbseq_pipeline.distance`)

Each dimension addresses an independent biological question; they are
intentionally preserved in their native, interpretable forms without collapsing
into an arbitrary scalar composite score.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


def build_perturbation_meta(
    cfg: Config,
    perturbation_table: Optional[pd.DataFrame] = None,
    ps_summary: Optional[pd.DataFrame] = None,
    lochness_summary: Optional[pd.DataFrame] = None,
    distance_table: Optional[pd.DataFrame] = None,
    cofunctional_modules: Optional[pd.DataFrame] = None,
    phenotype_modules: Optional[pd.DataFrame] = None,
    primary_control: str = "ntc",
) -> pd.DataFrame:
    """Merge all available target-level statistics into the master perturbation meta table.

    Parameters
    ----------
    cfg : Config
        Pipeline configuration.
    perturbation_table : Optional[pd.DataFrame]
        Target-level perturbation efficacy table (Stage 5).
    ps_summary : Optional[pd.DataFrame]
        Target-level PS score summary table (Stage 8).
    lochness_summary : Optional[pd.DataFrame]
        Target-level lochNESS summary table (Stage 9).
    distance_table : Optional[pd.DataFrame]
        Target-level perturbation distance table (Stage 10).
    cofunctional_modules : Optional[pd.DataFrame]
        Co-functional module assignments (Stage 7).
    phenotype_modules : Optional[pd.DataFrame]
        Phenotype-distance module assignments (Stage 11).
    primary_control : str
        Primary control group used for perturbation strength.

    Returns
    -------
    pd.DataFrame
        Master perturbation meta table with one row per target gene.
    """
    if not cfg.meta_analysis.enabled:
        logger.info("Meta-analysis table disabled (meta_analysis.enabled: false)")
        return pd.DataFrame()
    logger.info("=== Building Master Perturbation Meta Table ===")
    # Collect all unique targets across all available results
    target_sets: List[set] = []
    if perturbation_table is not None and not perturbation_table.empty and "target_gene" in perturbation_table.columns:
        target_sets.append(set(perturbation_table["target_gene"].dropna().astype(str)))
    if ps_summary is not None and not ps_summary.empty and "target_gene" in ps_summary.columns:
        target_sets.append(set(ps_summary["target_gene"].dropna().astype(str)))
    if lochness_summary is not None and not lochness_summary.empty and "target_gene" in lochness_summary.columns:
        target_sets.append(set(lochness_summary["target_gene"].dropna().astype(str)))
    if distance_table is not None and not distance_table.empty and "target_gene" in distance_table.columns:
        target_sets.append(set(distance_table["target_gene"].dropna().astype(str)))
    if cofunctional_modules is not None and not cofunctional_modules.empty:
        col = (
            "target_gene"
            if "target_gene" in cofunctional_modules.columns
            else (
                "target"
                if "target" in cofunctional_modules.columns
                else ("perturbation" if "perturbation" in cofunctional_modules.columns else None)
            )
        )
        if col:
            target_sets.append(set(cofunctional_modules[col].dropna().astype(str)))
    if phenotype_modules is not None and not phenotype_modules.empty and "target_gene" in phenotype_modules.columns:
        target_sets.append(set(phenotype_modules["target_gene"].dropna().astype(str)))
    if not target_sets:
        logger.warning("No target-level results available to assemble perturbation_meta.csv")
        return pd.DataFrame()
    all_targets = sorted(set().union(*target_sets))
    meta = pd.DataFrame({"target_gene": all_targets})
    # 1. Cell counts and Perturbation Efficacy (log2FC, FDR, Hit)
    if perturbation_table is not None and not perturbation_table.empty and "target_gene" in perturbation_table.columns:
        p_df = perturbation_table.copy().set_index("target_gene")
        # Cell count
        n_pert_col = next((c for c in ("n_perturbed", "n_cells", "n_perturbed_cells") if c in p_df.columns), None)
        if n_pert_col:
            meta["n_cells"] = meta["target_gene"].map(p_df[n_pert_col]).astype("Int64")
        # Efficacy log2FC
        lfc_col = (
            f"log2fc_{primary_control}"
            if f"log2fc_{primary_control}" in p_df.columns
            else ("log2fc" if "log2fc" in p_df.columns else None)
        )
        if lfc_col:
            meta["target_log2fc"] = meta["target_gene"].map(p_df[lfc_col]).astype(float)
        # Percent knockdown
        kd_col = (
            f"pct_knockdown_{primary_control}"
            if f"pct_knockdown_{primary_control}" in p_df.columns
            else ("pct_knockdown" if "pct_knockdown" in p_df.columns else None)
        )
        if kd_col:
            meta["target_pct_kd"] = meta["target_gene"].map(p_df[kd_col]).astype(float)
        # Efficacy FDR
        fdr_col = (
            f"ks_fdr_{primary_control}"
            if f"ks_fdr_{primary_control}" in p_df.columns
            else (
                f"mwu_fdr_{primary_control}"
                if f"mwu_fdr_{primary_control}" in p_df.columns
                else ("fdr" if "fdr" in p_df.columns else None)
            )
        )
        if fdr_col:
            meta["target_fdr"] = meta["target_gene"].map(p_df[fdr_col]).astype(float)
        # Hit calling
        hit_col = (
            f"is_hit_{primary_control}"
            if f"is_hit_{primary_control}" in p_df.columns
            else ("is_hit" if "is_hit" in p_df.columns else None)
        )
        if hit_col:
            meta["is_effective_hit"] = meta["target_gene"].map(p_df[hit_col]).astype("boolean")
    # 2. Single-Cell Perturbation Score (PS Score / Penetrance)
    if ps_summary is not None and not ps_summary.empty and "target_gene" in ps_summary.columns:
        ps_df = ps_summary.copy().set_index("target_gene")
        if "n_cells" not in meta.columns and "n_perturbed_cells" in ps_df.columns:
            meta["n_cells"] = meta["target_gene"].map(ps_df["n_perturbed_cells"]).astype("Int64")
        if "mean_ps" in ps_df.columns:
            meta["ps_mean"] = meta["target_gene"].map(ps_df["mean_ps"]).astype(float)
        if "median_ps" in ps_df.columns:
            meta["ps_median"] = meta["target_gene"].map(ps_df["median_ps"]).astype(float)
        if "pct_successful_kd" in ps_df.columns:
            meta["ps_responder_fraction"] = meta["target_gene"].map(ps_df["pct_successful_kd"]).astype(float)
        if "net_pct_kd" in ps_df.columns:
            meta["ps_net_responder_fraction"] = meta["target_gene"].map(ps_df["net_pct_kd"]).astype(float)
        if "pct_escaper" in ps_df.columns:
            meta["ps_escaper_fraction"] = meta["target_gene"].map(ps_df["pct_escaper"]).astype(float)
    # 3. lochNESS Continuous Manifold Enrichment (Topology)
    if lochness_summary is not None and not lochness_summary.empty and "target_gene" in lochness_summary.columns:
        loch_df = lochness_summary.copy().set_index("target_gene")
        if "mean_lochness_in_own_cells" in loch_df.columns:
            meta["lochness_mean"] = meta["target_gene"].map(loch_df["mean_lochness_in_own_cells"]).astype(float)
        elif "mean_lochness" in loch_df.columns:
            meta["lochness_mean"] = meta["target_gene"].map(loch_df["mean_lochness"]).astype(float)
        if "median_lochness_in_own_cells" in loch_df.columns:
            meta["lochness_median"] = meta["target_gene"].map(loch_df["median_lochness_in_own_cells"]).astype(float)
        if "max_lochness" in loch_df.columns:
            meta["lochness_peak"] = meta["target_gene"].map(loch_df["max_lochness"]).astype(float)
        if "pct_own_cells_enriched" in loch_df.columns:
            meta["lochness_pct_enriched"] = meta["target_gene"].map(loch_df["pct_own_cells_enriched"]).astype(float)
    # 4. Perturbation Distance & DistanceTest (Phenotype Magnitude)
    if distance_table is not None and not distance_table.empty and "target_gene" in distance_table.columns:
        d_df = distance_table.copy().set_index("target_gene")
        if "n_cells" not in meta.columns and "n_cells" in d_df.columns:
            meta["n_cells"] = meta["target_gene"].map(d_df["n_cells"]).astype("Int64")
        if "energy_distance" in d_df.columns:
            meta["energy_distance"] = meta["target_gene"].map(d_df["energy_distance"]).astype(float)
        if "mmd_distance" in d_df.columns:
            meta["mmd_distance"] = meta["target_gene"].map(d_df["mmd_distance"]).astype(float)
        if "pvalue" in d_df.columns:
            meta["distance_pvalue"] = meta["target_gene"].map(d_df["pvalue"]).astype(float)
        if "fdr" in d_df.columns:
            meta["distance_fdr"] = meta["target_gene"].map(d_df["fdr"]).astype(float)
        if "significant" in d_df.columns:
            meta["distance_significant"] = meta["target_gene"].map(d_df["significant"]).astype("boolean")
    # 5. Biological Organization (Co-functional & Phenotype Modules)
    if cofunctional_modules is not None and not cofunctional_modules.empty:
        c_df = cofunctional_modules.copy()
        t_col = (
            "target_gene"
            if "target_gene" in c_df.columns
            else (
                "target" if "target" in c_df.columns else ("perturbation" if "perturbation" in c_df.columns else None)
            )
        )
        m_col = (
            "module"
            if "module" in c_df.columns
            else ("cofunctional_module" if "cofunctional_module" in c_df.columns else None)
        )
        if t_col and m_col:
            c_map = c_df.set_index(t_col)[m_col]
            meta["cofunctional_module"] = meta["target_gene"].map(c_map).astype(str)
            meta.loc[meta["cofunctional_module"] == "nan", "cofunctional_module"] = None
    if phenotype_modules is not None and not phenotype_modules.empty and "target_gene" in phenotype_modules.columns:
        p_df = phenotype_modules.copy().set_index("target_gene")
        pm_col = (
            "phenotype_module"
            if "phenotype_module" in p_df.columns
            else ("module" if "module" in p_df.columns else None)
        )
        if pm_col:
            meta["phenotype_module"] = meta["target_gene"].map(p_df[pm_col]).astype(str)
            meta.loc[meta["phenotype_module"] == "nan", "phenotype_module"] = None
    # Sort primarily by energy distance if available, otherwise by target_gene
    if "energy_distance" in meta.columns and meta["energy_distance"].notna().any():
        meta = meta.sort_values("energy_distance", ascending=False).reset_index(drop=True)
    else:
        meta = meta.sort_values("target_gene").reset_index(drop=True)
    logger.info("Master perturbation meta table assembled: %d targets x %d features", len(meta), len(meta.columns))
    return meta
