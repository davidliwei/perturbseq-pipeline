"""Tests for metadata-aware guide-to-target mapping and backward compatibility."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from perturbseq_pipeline.config import Config, GuideConfig
from perturbseq_pipeline.guides import (
    CLASS_AMBIGUOUS,
    CLASS_NTC,
    CLASS_TARGETING,
    CLASS_UNASSIGNED,
    OBS_CLASS,
    OBS_GUIDE,
    OBS_TARGET,
    assign_guides,
    parse_target_genes,
    resolve_guide_targets,
    target_genes,
)
from perturbseq_pipeline.io import RAW_GUIDE_LABEL, write_guide_table


# Test A & B: Metadata target overrides guide parsing


def test_metadata_target_overrides_guide_parsing():
    """Authoritative metadata column provides biological target, not parsed guide ID."""
    guide_ids = ["TSS100020_17082653_23-ENST00000606659", "TSS100176_25368890_23-CUFF_46742_1"]
    target_names = ["CNOT7", "CDCA2"]
    # 2 cells, cell 0 has guide 0, cell 1 has guide 1
    X_guides = sparse.csr_matrix(np.array([[10.0, 0.0], [0.0, 10.0]]))
    var_guides = pd.DataFrame(
        {"target_gene_name": target_names, "target_gene_id": ["ENSG00000165140", "ENSG00000142188"]}, index=guide_ids
    )
    guide_adata = ad.AnnData(X=X_guides, var=var_guides)
    guide_adata.obs_names = ["cell_0", "cell_1"]
    expr_adata = ad.AnnData(
        X=sparse.csr_matrix(np.ones((2, 5))),
        obs=pd.DataFrame(index=["cell_0", "cell_1"]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
    )
    cfg = Config()
    cfg.guides.target_feature_column = "target_gene_name"
    cfg.guides.min_umi = 3
    cfg.guides.dominance_ratio = 2.0
    res = assign_guides(expr_adata, guide_adata, cfg)
    # Test A: TSS100020 -> CNOT7 and NOT TSS100020
    assert res.obs.loc["cell_0", OBS_TARGET] == "CNOT7"
    assert res.obs.loc["cell_0", OBS_TARGET] != "TSS100020"
    assert res.obs.loc["cell_0", OBS_CLASS] == CLASS_TARGETING
    assert res.obs.loc["cell_0", OBS_GUIDE] == "TSS100020_17082653_23-ENST00000606659"
    # Test B: TSS100176 -> CDCA2
    assert res.obs.loc["cell_1", OBS_TARGET] == "CDCA2"
    assert res.obs.loc["cell_1", OBS_TARGET] != "TSS100176"
    assert res.obs.loc["cell_1", OBS_CLASS] == CLASS_TARGETING


# Test C: Non-targeting metadata handling


def test_non_targeting_metadata():
    """Metadata target 'Non-Targeting' correctly collapses to configured NTC."""
    guide_ids = ["non-targeting_00300", "TSS100020_17082653_23-ENST00000606659"]
    target_names = ["Non-Targeting", "CNOT7"]
    X_guides = sparse.csr_matrix(np.array([[10.0, 0.0], [0.0, 10.0]]))
    var_guides = pd.DataFrame({"target_gene_name": target_names}, index=guide_ids)
    guide_adata = ad.AnnData(X=X_guides, var=var_guides)
    guide_adata.obs_names = ["cell_ntc", "cell_target"]
    expr_adata = ad.AnnData(
        X=sparse.csr_matrix(np.ones((2, 5))),
        obs=pd.DataFrame(index=["cell_ntc", "cell_target"]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
    )
    cfg = Config()
    cfg.guides.target_feature_column = "target_gene_name"
    cfg.guides.ntc_label = "ntc"
    res = assign_guides(expr_adata, guide_adata, cfg)
    assert res.obs.loc["cell_ntc", OBS_TARGET] == "ntc"
    assert res.obs.loc["cell_ntc", OBS_CLASS] == CLASS_NTC
    assert res.obs.loc["cell_ntc", OBS_GUIDE] == "non-targeting_00300"


# Test D: Ignored metadata handling


def test_ignored_metadata():
    """Metadata target 'Ignore' is mapped to unassigned, not a biological target."""
    guide_ids = ["guide_ignore", "guide_cnot7"]
    target_names = ["Ignore", "CNOT7"]
    X_guides = sparse.csr_matrix(np.array([[10.0, 0.0], [0.0, 10.0]]))
    var_guides = pd.DataFrame({"target_gene_name": target_names}, index=guide_ids)
    guide_adata = ad.AnnData(X=X_guides, var=var_guides)
    guide_adata.obs_names = ["cell_ignore", "cell_target"]
    expr_adata = ad.AnnData(
        X=sparse.csr_matrix(np.ones((2, 5))),
        obs=pd.DataFrame(index=["cell_ignore", "cell_target"]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
    )
    cfg = Config()
    cfg.guides.target_feature_column = "target_gene_name"
    cfg.guides.ignored_target_values = ["Ignore"]
    cfg.guides.unassigned_label = "unassigned"
    res = assign_guides(expr_adata, guide_adata, cfg)
    assert res.obs.loc["cell_ignore", OBS_TARGET] == "unassigned"
    assert res.obs.loc["cell_ignore", OBS_CLASS] == CLASS_UNASSIGNED
    assert res.obs.loc["cell_ignore", OBS_GUIDE] == "guide_ignore"
    # Confirm 'Ignore' does not appear in target genes
    targets = target_genes(res, cfg)
    assert "Ignore" not in targets
    assert "CNOT7" in targets


# Test E: Missing metadata column error


def test_missing_metadata_column_raises_value_error():
    """Configuring a non-existent metadata column fails with a clear ValueError."""
    guide_ids = ["guide_1", "guide_2"]
    X_guides = sparse.csr_matrix(np.ones((2, 2)))
    var_guides = pd.DataFrame({"other_col": ["A", "B"]}, index=guide_ids)
    guide_adata = ad.AnnData(X=X_guides, var=var_guides)
    guide_adata.obs_names = ["cell_1", "cell_2"]
    expr_adata = ad.AnnData(X=sparse.csr_matrix(np.ones((2, 2))), obs=pd.DataFrame(index=["cell_1", "cell_2"]))
    cfg = Config()
    cfg.guides.target_feature_column = "target_gene_name"
    with pytest.raises(ValueError, match="target_gene_name"):
        assign_guides(expr_adata, guide_adata, cfg)


# Test F: Backward compatibility with conventional guide-ID parsing


def test_backward_compatibility_guide_matrix_parsing():
    """When target_feature_column is null, legacy guide-ID parsing remains active."""
    guide_ids = ["AFF4_P1P2_1", "CD81_P1P2_2", "non-targeting_00300"]
    X_guides = sparse.csr_matrix(np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]))
    # var does not have target_feature_column, target_feature_column is None
    var_guides = pd.DataFrame(index=guide_ids)
    guide_adata = ad.AnnData(X=X_guides, var=var_guides)
    guide_adata.obs_names = ["cell_0", "cell_1", "cell_2"]
    expr_adata = ad.AnnData(
        X=sparse.csr_matrix(np.ones((3, 5))),
        obs=pd.DataFrame(index=["cell_0", "cell_1", "cell_2"]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
    )
    cfg = Config()
    assert cfg.guides.target_feature_column is None
    cfg.guides.ntc_label = "ntc"
    res = assign_guides(expr_adata, guide_adata, cfg)
    assert res.obs.loc["cell_0", OBS_TARGET] == "AFF4"
    assert res.obs.loc["cell_0", OBS_CLASS] == CLASS_TARGETING
    assert res.obs.loc["cell_1", OBS_TARGET] == "CD81"
    assert res.obs.loc["cell_1", OBS_CLASS] == CLASS_TARGETING
    assert res.obs.loc["cell_2", OBS_TARGET] == "ntc"
    assert res.obs.loc["cell_2", OBS_CLASS] == CLASS_NTC


# Test G: Backward compatibility with Replogle-style labels


def test_backward_compatibility_replogle_style_labels():
    """Replogle precomputed obs label path remains completely untouched."""
    expr_adata = ad.AnnData(
        X=sparse.csr_matrix(np.ones((4, 5))),
        obs=pd.DataFrame(
            {RAW_GUIDE_LABEL: ["AFF4", "CD81", "non-targeting", "unassigned"]}, index=[f"cell_{i}" for i in range(4)]
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
    )
    cfg = Config()
    cfg.guides.target_feature_column = "target_gene_name"  # Should NOT affect label path
    cfg.guides.ntc_label = "ntc"
    res = assign_guides(expr_adata, None, cfg)
    assert res.obs.loc["cell_0", OBS_TARGET] == "AFF4"
    assert res.obs.loc["cell_0", OBS_CLASS] == CLASS_TARGETING
    assert res.obs.loc["cell_1", OBS_TARGET] == "CD81"
    assert res.obs.loc["cell_1", OBS_CLASS] == CLASS_TARGETING
    assert res.obs.loc["cell_2", OBS_TARGET] == "ntc"
    assert res.obs.loc["cell_2", OBS_CLASS] == CLASS_NTC
    assert res.obs.loc["cell_3", OBS_TARGET] == "unassigned"
    assert res.obs.loc["cell_3", OBS_CLASS] == CLASS_UNASSIGNED


# Test H: Backward compatibility with KOLF-style labels


def test_backward_compatibility_kolf_style_labels():
    """KOLF precomputed obs gene_target path remains completely untouched."""
    expr_adata = ad.AnnData(
        X=sparse.csr_matrix(np.ones((4, 5))),
        obs=pd.DataFrame(
            {RAW_GUIDE_LABEL: ["DUX4", "BBC3", "NTC", "unassigned"]}, index=[f"cell_{i}" for i in range(4)]
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
    )
    cfg = Config()
    cfg.guides.ntc_patterns = ["^NTC$"]
    cfg.guides.ntc_label = "ntc"
    res = assign_guides(expr_adata, None, cfg)
    assert res.obs.loc["cell_0", OBS_TARGET] == "DUX4"
    assert res.obs.loc["cell_0", OBS_CLASS] == CLASS_TARGETING
    assert res.obs.loc["cell_1", OBS_TARGET] == "BBC3"
    assert res.obs.loc["cell_1", OBS_CLASS] == CLASS_TARGETING
    assert res.obs.loc["cell_2", OBS_TARGET] == "ntc"
    assert res.obs.loc["cell_2", OBS_CLASS] == CLASS_NTC
    assert res.obs.loc["cell_3", OBS_TARGET] == "unassigned"
    assert res.obs.loc["cell_3", OBS_CLASS] == CLASS_UNASSIGNED


# Test I: Configuration serialization and YAML roundtrip


def test_config_serialization_and_yaml_roundtrip(tmp_path):
    """Config serialization preserves target_feature_column and ignored_target_values."""
    cfg = Config()
    cfg.input.h5ad = "dummy.h5ad"
    cfg.guides.target_feature_column = "target_gene_name"
    cfg.guides.ignored_target_values = ["Ignore", "CUSTOM_IGNORE"]
    d = cfg.to_dict()
    assert d["guides"]["target_feature_column"] == "target_gene_name"
    assert d["guides"]["ignored_target_values"] == ["Ignore", "CUSTOM_IGNORE"]
    cfg2 = Config.from_dict(d)
    assert cfg2.guides.target_feature_column == "target_gene_name"
    assert cfg2.guides.ignored_target_values == ["Ignore", "CUSTOM_IGNORE"]
    yaml_path = tmp_path / "test_config.yaml"
    cfg.dump_yaml(yaml_path)
    loaded = Config.from_yaml(yaml_path)
    assert loaded.guides.target_feature_column == "target_gene_name"
    assert loaded.guides.ignored_target_values == ["Ignore", "CUSTOM_IGNORE"]
    # Test deserialization of legacy dict missing new fields
    legacy_dict = {"input": {"h5ad": "dummy.h5ad"}, "guides": {"min_umi": 5}}
    legacy_cfg = Config.from_dict(legacy_dict)
    assert legacy_cfg.guides.min_umi == 5
    assert legacy_cfg.guides.target_feature_column is None
    assert legacy_cfg.guides.ignored_target_values == ["Ignore"]


# Test J: Regression assertions


def test_regression_assertions():
    """Regression test ensuring TSS IDs and Ignore are excluded from biological targets."""
    guide_ids = [
        "TSS100020_17082653_23-ENST00000606659",
        "TSS100176_25368890_23-CUFF_46742_1",
        "TSS100380_ignore_guide",
        "non-targeting_00300",
    ]
    target_names = ["CNOT7", "CDCA2", "Ignore", "Non-Targeting"]
    X_guides = sparse.csr_matrix(
        np.array([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0], [0.0, 0.0, 10.0, 0.0], [0.0, 0.0, 0.0, 10.0]])
    )
    var_guides = pd.DataFrame({"target_gene_name": target_names}, index=guide_ids)
    guide_adata = ad.AnnData(X=X_guides, var=var_guides)
    guide_adata.obs_names = [f"cell_{i}" for i in range(4)]
    expr_adata = ad.AnnData(
        X=sparse.csr_matrix(np.ones((4, 5))),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(4)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(5)]),
    )
    cfg = Config()
    cfg.guides.target_feature_column = "target_gene_name"
    cfg.guides.ignored_target_values = ["Ignore"]
    cfg.guides.ntc_label = "ntc"
    res = assign_guides(expr_adata, guide_adata, cfg)
    assigned_targets = set(target_genes(res, cfg))
    # Mandatory regression assertions
    assert "TSS100020" not in assigned_targets
    assert "TSS100176" not in assigned_targets
    assert "TSS100380" not in assigned_targets
    assert "Ignore" not in assigned_targets
    assert "CNOT7" in assigned_targets
    assert "CDCA2" in assigned_targets
    assert "ntc" not in assigned_targets  # NTC is control, not in target_genes


# Test resolve_guide_targets helper directly


def test_resolve_guide_targets_direct_helper():
    """Unit test for resolve_guide_targets helper function."""
    var = pd.DataFrame(
        {"target_gene_name": ["  CNOT7  ", "CDCA2", "Ignore", "ignore", "Non-Targeting", "", None, np.nan]},
        index=[f"guide_{i}" for i in range(8)],
    )
    guides = ad.AnnData(X=sparse.csr_matrix((8, 8)), var=var)
    gcfg = GuideConfig(
        target_feature_column="target_gene_name", ignored_target_values=["Ignore"], unassigned_label="unassigned"
    )
    resolved = resolve_guide_targets(guides, gcfg)
    assert list(resolved) == [
        "CNOT7",
        "CDCA2",
        "unassigned",
        "unassigned",
        "Non-Targeting",
        "unassigned",
        "unassigned",
        "unassigned",
    ]
    # Test fallback when target_feature_column is None
    gcfg_none = GuideConfig(target_feature_column=None, target_split_delims=[])
    resolved_none = resolve_guide_targets(guides, gcfg_none)
    assert list(resolved_none) == [f"guide_{i}" for i in range(8)]


# Test write_guide_table with resolved metadata


def test_write_guide_table_preserves_resolved_metadata(tmp_path):
    """write_guide_table uses guides.var['target_gene'] when present."""
    guide_ids = ["TSS100020_1", "TSS100176_2"]
    X = sparse.csr_matrix(np.array([[5, 0], [0, 6]]))
    var = pd.DataFrame({"target_gene": ["CNOT7", "CDCA2"]}, index=guide_ids)
    guides = ad.AnnData(X=X, var=var)
    guides.obs_names = ["cell_1", "cell_2"]
    expr = ad.AnnData(
        X=sparse.csr_matrix(np.ones((2, 2))),
        obs=pd.DataFrame({OBS_TARGET: ["CNOT7", "CDCA2"]}, index=["cell_1", "cell_2"]),
    )
    cfg = Config()
    cfg.output.write_guide_table = True
    cfg.output.guide_table_min_umi = 3
    out_path = tmp_path / "guide_table.tsv"
    res_path = write_guide_table(guides, expr, cfg, out_path)
    assert res_path is not None
    table = pd.read_csv(res_path, sep="	")
    assert list(table["gene"]) == ["CNOT7", "CDCA2"]
    assert "TSS100020" not in list(table["gene"])


# Test Config validations


def test_config_validation_guide_target_column():
    """Config validation rejects empty target_feature_column or empty ignored values."""
    cfg = Config()
    cfg.input.h5ad = "dummy.h5ad"
    cfg.guides.target_feature_column = "   "
    with pytest.raises(ValueError, match="target_feature_column"):
        cfg.validate()
    cfg = Config()
    cfg.input.h5ad = "dummy.h5ad"
    cfg.guides.ignored_target_values = ["Ignore", "  "]
    with pytest.raises(ValueError, match="ignored_target_values"):
        cfg.validate()
