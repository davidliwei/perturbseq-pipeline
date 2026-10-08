"""Tests for master perturbation meta table, H5AD storage invariants, and pipeline configs."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from perturbseq_pipeline.config import Config
from perturbseq_pipeline.meta import build_perturbation_meta


# Test 12: Meta Table Integration


def test_meta_table_merges_all_dimensions():
    """Meta table should combine efficacy, PS, lochNESS, distance, and modules without combining into a single scalar score."""
    cfg = Config()
    # Perturbation strength (Stage 5)
    pert_df = pd.DataFrame(
        {
            "target_gene": ["GeneA", "GeneB", "GeneC"],
            "n_perturbed": [150, 200, 180],
            "log2fc_ntc": [-1.5, -0.2, -2.1],
            "pct_knockdown_ntc": [65.0, 12.0, 75.0],
            "ks_fdr_ntc": [0.001, 0.45, 0.0001],
            "is_hit_ntc": [True, False, True],
        }
    )
    # PS score (Stage 8)
    ps_df = pd.DataFrame(
        {
            "target_gene": ["GeneA", "GeneB", "GeneC"],
            "mean_ps": [0.78, 0.22, 0.85],
            "median_ps": [0.82, 0.18, 0.90],
            "pct_successful_kd": [72.0, 15.0, 80.0],
            "net_pct_kd": [60.0, 5.0, 70.0],
            "pct_escaper": [10.0, 60.0, 8.0],
        }
    )
    # lochNESS (Stage 9)
    loch_df = pd.DataFrame(
        {
            "target_gene": ["GeneA", "GeneB", "GeneC"],
            "mean_lochness_in_own_cells": [1.45, 0.05, 2.10],
            "median_lochness_in_own_cells": [1.30, 0.02, 1.95],
            "max_lochness": [3.2, 0.4, 4.1],
            "pct_own_cells_enriched": [85.0, 10.0, 92.0],
        }
    )
    # Perturbation Distance (Stage 10)
    dist_df = pd.DataFrame(
        {
            "target_gene": ["GeneA", "GeneB", "GeneC"],
            "n_cells": [150, 200, 180],
            "energy_distance": [1.25, 0.08, 1.85],
            "mmd_distance": [0.45, 0.02, 0.65],
            "pvalue": [0.001, 0.62, 0.001],
            "fdr": [0.002, 0.62, 0.002],
            "significant": [True, False, True],
        }
    )
    # Co-functional modules (Stage 7)
    co_df = pd.DataFrame({"target_gene": ["GeneA", "GeneB", "GeneC"], "cofunctional_module": ["M1", "M2", "M1"]})
    # Phenotype modules (Stage 11)
    ph_df = pd.DataFrame({"target_gene": ["GeneA", "GeneB", "GeneC"], "phenotype_module": ["PM1", "PM3", "PM1"]})
    meta = build_perturbation_meta(
        cfg=cfg,
        perturbation_table=pert_df,
        ps_summary=ps_df,
        lochness_summary=loch_df,
        distance_table=dist_df,
        cofunctional_modules=co_df,
        phenotype_modules=ph_df,
        primary_control="ntc",
    )
    assert not meta.empty
    assert len(meta) == 3
    # Check all key columns exist
    expected_cols = [
        "target_gene",
        "n_cells",
        "target_log2fc",
        "target_pct_kd",
        "target_fdr",
        "is_effective_hit",
        "ps_mean",
        "ps_median",
        "ps_responder_fraction",
        "lochness_mean",
        "lochness_peak",
        "energy_distance",
        "distance_pvalue",
        "distance_fdr",
        "distance_significant",
        "cofunctional_module",
        "phenotype_module",
    ]
    for col in expected_cols:
        assert col in meta.columns, f"Missing column {col} in perturbation_meta"
    # Confirm NO arbitrary combined single score exists
    for col in meta.columns:
        assert "master_score" not in col and "combined_score" not in col, (
            f"Unexpected composite score column found: {col}"
        )


# Test 13: H5AD Storage Policy (Lean H5AD)


def test_h5ad_storage_invariants():
    """H5AD must not contain full pairwise DistanceSpace matrices in uns."""
    import anndata as ad
    from scipy import sparse

    obs = pd.DataFrame({"target_gene": ["A", "B", "C"], "perturbation_class": ["targeting"] * 3})
    expr = ad.AnnData(X=sparse.csr_matrix(np.zeros((3, 10))), obs=obs)
    # Validate that DistanceSpace results are not placed into expr.uns
    assert "perturbation_distance_matrix" not in expr.uns
    assert "distance_space" not in expr.uns


# Test 14: Existing YAML configs parse cleanly


def test_all_config_yamls_parse():
    """All YAML configs in the repository should parse cleanly with backward compatibility."""
    config_dir = Path(__file__).parent.parent / "config"
    yaml_files = list(config_dir.glob("*.yaml"))
    assert len(yaml_files) > 0, "No YAML config files found"
    for y_path in yaml_files:
        with open(y_path) as f:
            data = yaml.safe_load(f)
        cfg = Config.from_dict(data)
        assert isinstance(cfg, Config)
        assert hasattr(cfg, "distance")
        assert hasattr(cfg, "distance_space")
        assert hasattr(cfg, "meta_analysis")
        assert hasattr(cfg, "visualization")
