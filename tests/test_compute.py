"""Unit tests for the CPU/GPU-aware compute backend execution layer."""

import os
from pathlib import Path
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from perturbseq_pipeline.compute import (
    ComputeDecision,
    ComputeProfiler,
    derive_seed,
    detect_available_cpus,
    detect_slurm_cpus,
    estimate_gpu_memory_safe,
    get_gpu_memory_info,
    is_gpu_available,
    is_package_available,
    limit_blas_threads,
    log_compute_decision,
    resolve_stage_backend,
    resolve_worker_count,
    run_parallel,
    stage_profile,
)
from perturbseq_pipeline.config import ComputeConfig, Config
from perturbseq_pipeline.distance import compute_perturbation_distance
from perturbseq_pipeline.guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS, OBS_GUIDE, OBS_TARGET
from perturbseq_pipeline.perturbation import test_all_targets as run_test_all_targets


# 1. CPU & SLURM Detection


def test_detect_slurm_cpus_parsing():
    """SLURM environment variables should be correctly parsed."""
    with mock.patch.dict(os.environ, {}, clear=True):
        assert detect_slurm_cpus() is None
    with mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "8"}, clear=True):
        assert detect_slurm_cpus() == 8
    with mock.patch.dict(os.environ, {"SLURM_CPUS_ON_NODE": "16"}, clear=True):
        assert detect_slurm_cpus() == 16
    with mock.patch.dict(os.environ, {"SLURM_JOB_CPUS_PER_NODE": "4(x2),8"}, clear=True):
        assert detect_slurm_cpus() == 4


def test_detect_available_cpus_respects_slurm():
    """detect_available_cpus should prefer SLURM allocation over physical count."""
    with mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "6"}, clear=True):
        assert detect_available_cpus() == 6


def test_resolve_worker_count():
    """Worker counts should respect stage overrides and available CPU limits."""
    cfg = Config()
    cfg.compute.n_jobs = 8
    cfg.compute.distance_n_jobs = 4
    with mock.patch("perturbseq_pipeline.compute.detect_available_cpus", return_value=16):
        # Default stage uses global n_jobs
        assert resolve_worker_count(cfg, "perturbation") == 8
        # Distance stage uses stage override
        assert resolve_worker_count(cfg, "distance") == 4
    # If available CPUs is lower, it should cap
    with mock.patch("perturbseq_pipeline.compute.detect_available_cpus", return_value=2):
        assert resolve_worker_count(cfg, "perturbation") == 2
        assert resolve_worker_count(cfg, "distance") == 2


# 2. GPU Detection & Memory Safety


def test_is_gpu_available_mock():
    """Test GPU detection helper with mock PyTorch/CuPy."""
    with mock.patch("perturbseq_pipeline.compute.is_package_available", return_value=True):
        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            assert is_gpu_available() is True
    with (
        mock.patch("perturbseq_pipeline.compute.is_package_available", return_value=False),
        mock.patch.dict("sys.modules", {"torch": None, "cupy": None, "pynvml": None}),
    ):
        assert is_gpu_available() is False


def test_gpu_memory_safety_check():
    """estimate_gpu_memory_safe should enforce fraction threshold."""
    mock_info = (8 * (1024**3), 16 * (1024**3))  # 8 GB free, 16 GB total
    with mock.patch("perturbseq_pipeline.compute.get_gpu_memory_info", return_value=mock_info):
        # Requesting 2 GB with 0.8 fraction (max allowed: 0.8 * 8 GB = 6.4 GB) -> Safe
        assert estimate_gpu_memory_safe(2 * (1024**3), fraction=0.8) is True
        # Requesting 7 GB -> Unsafe
        assert estimate_gpu_memory_safe(7 * (1024**3), fraction=0.8) is False


# 3. Deterministic Backend Decision Resolution


def test_resolve_stage_backend_cpu_mode():
    """When compute.backend='cpu', decision must always be CPU."""
    cfg = Config()
    cfg.compute.backend = "cpu"
    cfg.compute.n_jobs = 4
    decision = resolve_stage_backend("clustering", cfg, n_cells=500_000)
    assert decision.backend == "cpu"
    assert decision.is_gpu is False
    assert decision.n_jobs <= 4


def test_resolve_stage_backend_gpu_fallback():
    """When compute.backend='gpu' but GPU is unavailable, fallback to CPU."""
    cfg = Config()
    cfg.compute.backend = "gpu"
    cfg.compute.n_jobs = 4
    with mock.patch("perturbseq_pipeline.compute.is_gpu_available", return_value=False):
        decision = resolve_stage_backend("clustering", cfg, n_cells=10_000)
        assert decision.backend == "cpu"
        assert decision.is_gpu is False
        assert "falling back" in decision.reason.lower() or "not detected" in decision.reason.lower()


def test_resolve_stage_backend_auto_mode():
    """In AUTO mode, use CPU for small data and GPU for large data when eligible."""
    cfg = Config()
    cfg.compute.backend = "auto"
    cfg.compute.gpu_min_cells = 200_000
    # Small cell count -> CPU
    decision_small = resolve_stage_backend("clustering", cfg, n_cells=50_000)
    assert decision_small.backend == "cpu"
    assert decision_small.is_gpu is False
    # Large cell count with GPU available and mock memory safe
    with (
        mock.patch("perturbseq_pipeline.compute.is_gpu_available", return_value=True),
        mock.patch("perturbseq_pipeline.compute.estimate_gpu_memory_safe", return_value=True),
        mock.patch("perturbseq_pipeline.compute.is_package_available", return_value=True),
    ):
        decision_large = resolve_stage_backend("clustering", cfg, n_cells=300_000)
        assert decision_large.backend == "gpu"
        assert decision_large.is_gpu is True


# 4. Seed Derivation & BLAS Thread Limits


def test_derive_seed_deterministic():
    """derive_seed must be deterministic and vary across target names."""
    s1 = derive_seed(1234, "MYC")
    s2 = derive_seed(1234, "MYC")
    s3 = derive_seed(1234, "TP53")
    s4 = derive_seed(5678, "MYC")
    assert s1 == s2
    assert s1 != s3
    assert s1 != s4
    assert 0 <= s1 < (2**31 - 1)


def test_limit_blas_threads_context():
    """limit_blas_threads context manager executes without error."""
    with limit_blas_threads(1):
        A = np.ones((50, 50))
        B = np.dot(A, A)
        assert B.shape == (50, 50)


# 5. Multiprocessing Parallel vs Serial Execution


def _square_worker(x: int) -> int:
    return x * x


def test_run_parallel_equivalence():
    """run_parallel should give identical results between serial and parallel execution."""
    tasks = list(range(20))
    res_serial = run_parallel(_square_worker, tasks, n_jobs=1)
    res_parallel = run_parallel(_square_worker, tasks, n_jobs=2)
    assert res_serial == res_parallel
    assert res_parallel == [x * x for x in tasks]


# 6. Profiler & Benchmarking


def test_compute_profiler_records_and_exports(tmp_path: Path):
    """ComputeProfiler records stage elapsed time and writes CSV."""
    profiler = ComputeProfiler()
    with stage_profile("test_stage", profiler):
        # Small computation
        _ = sum(i * i for i in range(10_000))
    df = profiler.to_dataframe()
    assert not df.empty
    assert "stage" in df.columns
    assert "elapsed_seconds" in df.columns
    assert "peak_rss_mb" in df.columns
    assert df.iloc[0]["stage"] == "test_stage"
    csv_out = tmp_path / "compute_profile.csv"
    profiler.save_csv(csv_out)
    assert csv_out.exists()
    loaded_df = pd.read_csv(csv_out)
    assert len(loaded_df) == 1
    assert loaded_df.iloc[0]["stage"] == "test_stage"


# 7. Config Validation for Compute Fields


def test_compute_config_validation():
    """Compute configuration validation should reject invalid values."""
    cfg = Config()
    cfg.input.h5ad = "dummy.h5ad"
    cfg.validate()  # Default should pass
    # Invalid backend
    cfg_bad_backend = Config()
    cfg_bad_backend.input.h5ad = "dummy.h5ad"
    cfg_bad_backend.compute.backend = "quantum"
    with pytest.raises(ValueError, match="compute.backend"):
        cfg_bad_backend.validate()
    # Invalid n_jobs
    cfg_bad_jobs = Config()
    cfg_bad_jobs.input.h5ad = "dummy.h5ad"
    cfg_bad_jobs.compute.n_jobs = 0
    with pytest.raises(ValueError, match="compute.n_jobs"):
        cfg_bad_jobs.validate()
    # Invalid memory fraction
    cfg_bad_frac = Config()
    cfg_bad_frac.input.h5ad = "dummy.h5ad"
    cfg_bad_frac.compute.gpu_memory_fraction = 1.5
    with pytest.raises(ValueError, match="compute.gpu_memory_fraction"):
        cfg_bad_frac.validate()


# 8. Biological Stage Serial vs Parallel Exact Equivalence


@pytest.fixture
def synthetic_perturbation_adata():
    """Synthetic AnnData for testing perturbation strength and distance."""
    rng = np.random.default_rng(42)
    n_cells = 300
    n_genes = 20
    X = rng.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(np.float32)
    var_names = [f"gene_{i}" for i in range(n_genes)]
    obs_names = [f"cell_{i}" for i in range(n_cells)]
    # Assign perturbations
    targets = ["gene_0", "gene_1", "gene_2", "non-targeting"]
    assigned_targets = rng.choice(targets, size=n_cells, p=[0.25, 0.25, 0.25, 0.25])
    assigned_classes = [CLASS_NTC if t == "non-targeting" else CLASS_TARGETING for t in assigned_targets]
    obs = pd.DataFrame(
        {
            OBS_TARGET: assigned_targets,
            OBS_CLASS: assigned_classes,
            OBS_GUIDE: [f"guide_{t}" for t in assigned_targets],
            "lane_id": ["lane1"] * n_cells,
        },
        index=obs_names,
    )
    adata = ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=pd.DataFrame(index=var_names))
    adata.layers["counts"] = adata.X.copy()
    adata.layers["lognorm"] = adata.X.copy()
    adata.obsm["X_pca"] = rng.normal(size=(n_cells, 10)).astype(np.float64)
    return adata


def test_perturbation_serial_vs_parallel_equivalence(synthetic_perturbation_adata):
    """Perturbation testing must yield identical results across worker counts."""
    adata = synthetic_perturbation_adata
    cfg_serial = Config()
    cfg_serial.compute.n_jobs = 1
    cfg_serial.perturbation.min_cells_per_target = 5
    cfg_serial.perturbation.min_control_cells = 5
    cfg_parallel = Config()
    cfg_parallel.compute.n_jobs = 2
    cfg_parallel.perturbation.min_cells_per_target = 5
    cfg_parallel.perturbation.min_control_cells = 5
    res_serial = run_test_all_targets(adata, cfg_serial)
    res_parallel = run_test_all_targets(adata, cfg_parallel)
    pd.testing.assert_frame_equal(res_serial.table, res_parallel.table)


def test_distance_serial_vs_parallel_equivalence(synthetic_perturbation_adata):
    """Perturbation distance testing must yield identical results across worker counts."""
    adata = synthetic_perturbation_adata
    cfg_serial = Config()
    cfg_serial.distance.enabled = True  # optional stage, off by default
    cfg_serial.compute.n_jobs = 1
    cfg_serial.distance.min_cells = 5
    cfg_serial.distance.n_permutations = 50
    cfg_serial.distance.random_seed = 42
    cfg_parallel = Config()
    cfg_parallel.distance.enabled = True
    cfg_parallel.compute.n_jobs = 2
    cfg_parallel.distance.min_cells = 5
    cfg_parallel.distance.n_permutations = 50
    cfg_parallel.distance.random_seed = 42
    res_serial = compute_perturbation_distance(adata, cfg_serial)
    res_parallel = compute_perturbation_distance(adata, cfg_parallel)
    assert res_serial is not None and res_parallel is not None
    pd.testing.assert_frame_equal(res_serial.table, res_parallel.table)
