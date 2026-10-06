"""Unit tests for storage configuration and centralized data access layer."""

from pathlib import Path
import tempfile
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from perturbseq_pipeline.config import Config, StorageConfig
from perturbseq_pipeline.data_access import (
    SharedArrayBuffer,
    get_control_indices,
    get_embedding,
    get_expression_vector,
    get_layer,
    get_obs_column,
    get_target_indices_map,
    is_backed,
    log_storage_diagnostics,
    resolve_storage_mode,
)
from perturbseq_pipeline.guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS, OBS_TARGET


# 1. StorageConfig Defaults and Validation


def test_storage_config_defaults():
    """Verify default values for StorageConfig."""
    sc = StorageConfig()
    assert sc.mode == "auto"
    assert sc.backed_threshold_cells == 1_000_000
    assert sc.prefer_backed_h5ad is True
    assert sc.keep_embeddings_in_memory is True
    assert sc.shared_worker_arrays is True


def test_storage_config_in_config():
    """Verify Config initializes default StorageConfig."""
    cfg = Config()
    assert hasattr(cfg, "storage")
    assert isinstance(cfg.storage, StorageConfig)
    assert cfg.storage.mode == "auto"


def test_storage_config_validation():
    """Verify invalid storage mode and threshold rejection."""
    cfg = Config()
    cfg.input.h5ad = "dummy.h5ad"
    cfg.storage.mode = "invalid_mode"
    with pytest.raises(ValueError, match="storage.mode must be one of"):
        cfg.validate()
    cfg.storage.mode = "auto"
    cfg.storage.backed_threshold_cells = 0
    with pytest.raises(ValueError, match="storage.backed_threshold_cells must be >= 1"):
        cfg.validate()


def test_resolve_storage_mode():
    """Verify resolve_storage_mode helper under various configurations."""
    cfg = Config()
    # Default auto mode
    assert resolve_storage_mode(cfg, n_cells=100_000, is_h5ad_input=True) == "in_memory"
    assert resolve_storage_mode(cfg, n_cells=1_500_000, is_h5ad_input=True) == "backed"
    assert resolve_storage_mode(cfg, n_cells=1_500_000, is_h5ad_input=False) == "in_memory"
    # Forced in_memory
    cfg.storage.mode = "in_memory"
    assert resolve_storage_mode(cfg, n_cells=2_000_000, is_h5ad_input=True) == "in_memory"
    # Forced backed
    cfg.storage.mode = "backed"
    assert resolve_storage_mode(cfg, n_cells=100, is_h5ad_input=True) == "backed"
    assert resolve_storage_mode(cfg, n_cells=100, is_h5ad_input=False) == "in_memory"


# 2. Data Access Layer Helpers with AnnData


@pytest.fixture
def sample_adata():
    """Create a small AnnData object with sparse X, layers, obsm, and obs."""
    n_cells = 100
    n_genes = 20
    rng = np.random.default_rng(42)
    X_dense = rng.poisson(lam=1.5, size=(n_cells, n_genes)).astype(np.float32)
    X_sparse = sparse.csr_matrix(X_dense)
    var_names = [f"Gene_{i}" for i in range(n_genes)]
    obs_names = [f"Cell_{i}" for i in range(n_cells)]
    obs = pd.DataFrame(
        {
            OBS_TARGET: ["TargetA"] * 40 + ["TargetB"] * 40 + ["non-targeting"] * 20,
            OBS_CLASS: [CLASS_TARGETING] * 80 + [CLASS_NTC] * 20,
            "lane_id": ["L1"] * 50 + ["L2"] * 50,
        },
        index=obs_names,
    )
    adata = ad.AnnData(X=X_sparse, obs=obs, var=pd.DataFrame(index=var_names))
    adata.layers["lognorm"] = X_sparse.copy()
    adata.obsm["X_pca"] = rng.normal(size=(n_cells, 10)).astype(np.float32)
    return adata


def test_get_embedding(sample_adata):
    """Verify get_embedding extracts array with proper dtype and cell slicing."""
    emb = get_embedding(sample_adata, rep_name="X_pca")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (100, 10)
    assert emb.dtype == np.float32
    # Sliced extraction
    idx = np.array([0, 5, 10], dtype=np.int64)
    emb_sub = get_embedding(sample_adata, rep_name="X_pca", cell_indices=idx)
    assert emb_sub.shape == (3, 10)
    np.testing.assert_allclose(emb_sub, emb[idx])


def test_get_expression_vector(sample_adata):
    """Verify get_expression_vector retrieves 1D array without densifying full matrix."""
    vec = get_expression_vector(sample_adata, "Gene_0")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (100,)
    # Sliced extraction
    idx = np.array([2, 4, 6], dtype=np.int64)
    vec_sub = get_expression_vector(sample_adata, "Gene_0", cell_indices=idx)
    assert vec_sub.shape == (3,)
    np.testing.assert_allclose(vec_sub, vec[idx])
    # Missing gene raises KeyError
    with pytest.raises(KeyError):
        get_expression_vector(sample_adata, "NonexistentGene")


def test_get_target_indices_map(sample_adata):
    """Verify get_target_indices_map precomputes target to cell index mapping."""
    target_map = get_target_indices_map(sample_adata)
    assert "TargetA" in target_map
    assert "TargetB" in target_map
    assert len(target_map["TargetA"]) == 40
    assert len(target_map["TargetB"]) == 40
    assert np.all(target_map["TargetA"] == np.arange(40))
    assert np.all(target_map["TargetB"] == np.arange(40, 80))


def test_get_control_indices(sample_adata):
    """Verify get_control_indices returns NTC indices."""
    cfg = Config()
    ntc_idx = get_control_indices(sample_adata, cfg, control_choice="ntc")
    assert len(ntc_idx) == 20
    assert np.all(ntc_idx == np.arange(80, 100))


def test_shared_array_buffer():
    """Verify SharedArrayBuffer creation, memory mapping, and cleanup."""
    arr = np.arange(1000, dtype=np.float32).reshape(100, 10)
    buf = SharedArrayBuffer(arr, create_memmap=True)
    assert buf.shape == (100, 10)
    assert np.allclose(buf.array, arr)
    temp_path = buf._temp_file
    assert temp_path is not None and temp_path.exists()
    buf.close()
    assert not temp_path.exists()
