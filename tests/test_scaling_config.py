"""Unit tests for centralized dataset scaling configuration (ScalingConfig)."""

from pathlib import Path
import pytest
import yaml

from perturbseq_pipeline.config import Config, ScalingConfig


def test_scaling_config_defaults():
    """Verify default values for ScalingConfig match pipeline specifications."""
    sc = ScalingConfig()
    assert sc.mode == "auto"
    assert sc.large_n_cells == 1_000_000
    assert sc.large_n_perturbations == 5_000
    assert sc.marker_max_cells == 200_000
    assert sc.effect_gene_chunk == 256
    assert sc.guide_chunk_size == 20_000
    assert sc.guide_max_dense_elements == 20_000_000
    assert sc.collect_between_stages is True
    assert sc.log_memory is True
    assert sc.report_preview_rows == 500


def test_scaling_config_in_base_config():
    """Verify Config automatically initializes default ScalingConfig."""
    cfg = Config()
    assert hasattr(cfg, "scaling")
    assert isinstance(cfg.scaling, ScalingConfig)
    assert cfg.scaling.mode == "auto"


def test_use_large_mode_auto_replogle_scale():
    """Normal screen scale (~310k cells, ~2k perturbations) must stay standard."""
    cfg = Config()
    # Replogle scale: 310,385 cells x 2,057 perturbations
    assert not cfg.use_large_mode(310_385, n_perturbations=2_057)
    assert cfg.execution_mode(310_385, n_perturbations=2_057) == "standard"
    assert not cfg.use_large_mode(310_385)
    assert cfg.execution_mode(310_385) == "standard"


def test_use_large_mode_auto_kolf_scale():
    """Very large screen scale (~2.66m cells, ~11k perturbations) must enter large mode."""
    cfg = Config()
    # KOLF scale: 2,659,209 cells x 11,686 perturbations
    assert cfg.use_large_mode(2_659_209, n_perturbations=11_686)
    assert cfg.execution_mode(2_659_209, n_perturbations=11_686) == "large"
    assert cfg.use_large_mode(2_659_209)
    assert cfg.execution_mode(2_659_209) == "large"


def test_use_large_mode_auto_threshold_triggers():
    """Auto mode should trigger if either cell or perturbation threshold is met."""
    cfg = Config()
    # Cells >= 1_000_000
    assert cfg.use_large_mode(1_000_000, n_perturbations=100)
    assert cfg.execution_mode(1_000_000, n_perturbations=100) == "large"
    assert not cfg.use_large_mode(999_999, n_perturbations=100)
    # Perturbations >= 5_000 (even with small cell count)
    assert cfg.use_large_mode(50_000, n_perturbations=5_000)
    assert cfg.execution_mode(50_000, n_perturbations=5_000) == "large"
    assert not cfg.use_large_mode(50_000, n_perturbations=4_999)


def test_use_large_mode_forced_large():
    """Explicit mode='large' forces large mode regardless of dataset size."""
    cfg = Config()
    cfg.scaling.mode = "large"
    assert cfg.use_large_mode(10, n_perturbations=2)
    assert cfg.execution_mode(10, n_perturbations=2) == "large"
    assert cfg.use_large_mode(10)


def test_use_large_mode_forced_standard():
    """Explicit mode='standard' forces standard mode regardless of dataset size."""
    cfg = Config()
    cfg.scaling.mode = "standard"
    assert not cfg.use_large_mode(5_000_000, n_perturbations=20_000)
    assert cfg.execution_mode(5_000_000, n_perturbations=20_000) == "standard"
    assert not cfg.use_large_mode(5_000_000)


def test_scaling_config_yaml_loading(tmp_path: Path):
    """Verify parsing scaling settings from a YAML file."""
    yaml_content = """
run:
  name: test_run
input:
  h5ad: dummy.h5ad
scaling:
  mode: large
  large_n_cells: 500000
  large_n_perturbations: 2000
  marker_max_cells: 100000
  effect_gene_chunk: 512
  guide_chunk_size: 10000
  guide_max_dense_elements: 10000000
  collect_between_stages: false
  log_memory: false
  report_preview_rows: 250
"""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml_content)
    cfg = Config.from_yaml(cfg_file)
    assert cfg.scaling.mode == "large"
    assert cfg.scaling.large_n_cells == 500_000
    assert cfg.scaling.large_n_perturbations == 2_000
    assert cfg.scaling.marker_max_cells == 100_000
    assert cfg.scaling.effect_gene_chunk == 512
    assert cfg.scaling.guide_chunk_size == 10_000
    assert cfg.scaling.guide_max_dense_elements == 10_000_000
    assert cfg.scaling.collect_between_stages is False
    assert cfg.scaling.log_memory is False
    assert cfg.scaling.report_preview_rows == 250


def test_scaling_config_validation_invalid_mode():
    """Validation must reject unrecognized scaling modes."""
    cfg = Config()
    cfg.input.h5ad = "dummy.h5ad"
    cfg.scaling.mode = "invalid_mode"
    with pytest.raises(ValueError, match="scaling.mode must be"):
        cfg.validate()


@pytest.mark.parametrize(
    "field,invalid_val,msg",
    [
        ("large_n_cells", 0, "scaling.large_n_cells must be >= 1"),
        ("large_n_cells", -5, "scaling.large_n_cells must be >= 1"),
        ("large_n_perturbations", 0, "scaling.large_n_perturbations must be >= 1"),
        ("marker_max_cells", 0, "scaling.marker_max_cells must be >= 1"),
        ("effect_gene_chunk", 0, "scaling.effect_gene_chunk must be >= 1"),
        ("guide_chunk_size", 0, "scaling.guide_chunk_size must be >= 1"),
        ("guide_max_dense_elements", 0, "scaling.guide_max_dense_elements must be >= 1"),
        ("report_preview_rows", 0, "scaling.report_preview_rows must be >= 1"),
    ],
)
def test_scaling_config_validation_numeric_bounds(field, invalid_val, msg):
    """Validation must enforce strictly positive integer parameters."""
    cfg = Config()
    cfg.input.h5ad = "dummy.h5ad"
    setattr(cfg.scaling, field, invalid_val)
    with pytest.raises(ValueError, match=msg):
        cfg.validate()


def test_scaling_config_unknown_yaml_key(tmp_path: Path):
    """Validation must reject unknown keys in scaling block."""
    yaml_content = """
run:
  name: test_run
input:
  h5ad: dummy.h5ad
scaling:
  mode: auto
  non_existent_scaling_key: 123
"""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml_content)
    with pytest.raises(ValueError, match="Unknown config key.*non_existent_scaling_key"):
        Config.from_yaml(cfg_file)
