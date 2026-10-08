"""Compute backend abstraction, hardware detection, CPU multiprocessing, and GPU safety.

This module provides a centralized compute execution layer for perturbseq-pipeline:

1. **Hardware & Environment Detection**:
   - Detects logical and physical CPUs, Linux affinity, and SLURM allocations
     (``SLURM_CPUS_PER_TASK``, etc.).
   - Prevents oversubscription on shared nodes.
   - Detects GPU hardware availability and optional CUDA acceleration libraries.

2. **Stage Backend Decision**:
   - Centralizes CPU vs GPU placement per stage based on ``compute.backend``,
     dataset scale, dense representation sizes, and memory safety checks.
   - Gracefully falls back to CPU if GPU memory is insufficient or libraries are missing.

3. **CPU Multiprocessing Framework**:
   - Target-parallel execution helper with deterministic per-target seeds.
   - Controls internal BLAS/OpenMP threading per worker via ``threadpoolctl`` to
     prevent nested thread explosion.
   - Prevents serializing or duplicating full AnnData objects to workers.

4. **Compute Profiler & Benchmarking**:
   - Records elapsed time, backend, worker count, and peak memory per stage.
   - Writes ``tables/compute_profile.csv``.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib.util
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("perturbseq_pipeline.compute")


# Hardware & Environment Detection


def detect_slurm_cpus() -> Optional[int]:
    """Detect CPU allocation from SLURM environment variables if present.

    Prefers ``SLURM_CPUS_PER_TASK``, followed by ``SLURM_JOB_CPUS_PER_NODE``
    or ``SLURM_CPUS_ON_NODE``.
    """
    if "SLURM_CPUS_PER_TASK" in os.environ:
        try:
            val = int(os.environ["SLURM_CPUS_PER_TASK"].strip())
            if val > 0:
                return val
        except (ValueError, TypeError):
            pass
    for var in ("SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
        if var in os.environ:
            raw = os.environ[var].strip()
            match = re.match(r"^(\d+)", raw)
            if match:
                try:
                    val = int(match.group(1))
                    if val > 0:
                        return val
                except (ValueError, TypeError):
                    pass
    return None


def detect_available_cpus() -> int:
    """Detect the maximum usable CPU count for the current process/job.

    Respects SLURM allocation first, then process CPU affinity (e.g. cgroups/containers),
    and falls back to system CPU count. Never exceeds scheduler limits.
    """
    slurm_cpus = detect_slurm_cpus()
    if slurm_cpus is not None and slurm_cpus > 0:
        return slurm_cpus
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = os.sched_getaffinity(0)
            if affinity:
                return len(affinity)
        except Exception:
            pass
    return max(1, os.cpu_count() or 1)


def resolve_worker_count(configured_n_jobs: Any, stage_n_jobs: Optional[Any] = None) -> int:
    """Determine effective deterministic CPU worker count.

    Accepts either (cfg, stage_name) or (configured_n_jobs, stage_n_jobs).
    Per-stage override takes precedence if supplied. If value is -1,
    resolves to all available CPUs according to scheduler/system limits.
    """
    if hasattr(configured_n_jobs, "compute"):
        cfg = configured_n_jobs
        stage_name = str(stage_n_jobs) if stage_n_jobs is not None else ""
        comp = cfg.compute
        stage_override = None
        if stage_name in ("distance", "distance_space") and comp.distance_n_jobs is not None:
            stage_override = comp.distance_n_jobs
        elif stage_name == "perturbation" and comp.perturbation_n_jobs is not None:
            stage_override = comp.perturbation_n_jobs
        elif stage_name == "enrichment" and comp.enrichment_n_jobs is not None:
            stage_override = comp.enrichment_n_jobs
        elif stage_name == "modules" and comp.modules_n_jobs is not None:
            stage_override = comp.modules_n_jobs
        elif stage_name == "lochness" and comp.lochness_n_jobs is not None:
            stage_override = comp.lochness_n_jobs
        req = stage_override if stage_override is not None else comp.n_jobs
    else:
        req = stage_n_jobs if stage_n_jobs is not None else configured_n_jobs
    avail = detect_available_cpus()
    if req == -1:
        return max(1, avail)
    if req < 1:
        return 1
    return max(1, min(req, avail))


def is_package_available(package_name: str) -> bool:
    """Check if an optional Python package is installed and importable."""
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except Exception:
        return False


def is_gpu_available() -> bool:
    """Check whether a usable CUDA GPU is available on the system."""
    # 1. PyTorch check
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return True
    except Exception:
        pass
    # 2. CuPy check
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            return True
    except Exception:
        pass
    # 3. pynvml / nvidia-smi device check
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        if count > 0:
            return True
    except Exception:
        pass
    return False


def get_gpu_memory_info(device: int = 0) -> Tuple[int, int]:
    """Return ``(free_bytes, total_bytes)`` for the specified GPU device."""
    # 1. Try PyTorch
    try:
        import torch

        if torch.cuda.is_available() and device < torch.cuda.device_count():
            free_b, total_b = torch.cuda.mem_get_info(device)
            return int(free_b), int(total_b)
    except Exception:
        pass
    # 2. Try CuPy
    try:
        import cupy as cp

        if device < cp.cuda.runtime.getDeviceCount():
            with cp.cuda.Device(device):
                free_b, total_b = cp.cuda.runtime.memGetInfo()
                return int(free_b), int(total_b)
    except Exception:
        pass
    # 3. Try pynvml
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_b, total_b = mem_info.free, mem_info.total
        pynvml.nvmlShutdown()
        return int(free_b), int(total_b)
    except Exception:
        pass
    return 0, 0


def estimate_gpu_memory_safe(estimated_bytes: int, device: int = 0, fraction: float = 0.80) -> bool:
    """Check if the estimated byte footprint fits safely in free GPU memory."""
    free_b, total_b = get_gpu_memory_info(device)
    if total_b <= 0 or free_b <= 0:
        return False
    return estimated_bytes <= int(free_b * fraction)


# Backend Decision Layer


@dataclass
class ComputeDecision:
    """Resolved compute backend placement for a specific pipeline stage."""

    stage: str
    backend: str  # 'cpu' | 'gpu'
    n_jobs: int
    reason: str
    device: int = 0

    @property
    def is_gpu(self) -> bool:
        return self.backend.lower() == "gpu"

    @property
    def is_cpu(self) -> bool:
        return self.backend.lower() == "cpu"


def resolve_stage_backend(
    stage: str, cfg: Any, n_cells: int = 0, extra_info: Optional[Dict[str, Any]] = None
) -> ComputeDecision:
    """Resolve CPU vs GPU execution backend and worker count for a stage.

    Parameters
    ----------
    stage : str
        Name of the stage: 'clustering', 'perturbation', 'enrichment',
        'modules_effect', 'modules_correlation', 'modules_scoring',
        'ps_score', 'lochness', 'distance', 'distance_space'.
    cfg : Config
        Pipeline configuration containing compute settings.
    n_cells : int
        Number of cells in the active dataset.
    extra_info : Optional[Dict[str, Any]]
        Optional context (e.g. matrix element count, shape, etc.).

    Returns
    -------
    ComputeDecision
        Resolved placement decision.
    """
    comp = cfg.compute
    req_backend = comp.backend.lower()
    device = comp.gpu_device
    extra_info = extra_info or {}
    # Check stage-specific n_jobs override
    stage_override_attr = f"{stage}_n_jobs"
    stage_n_jobs = getattr(comp, stage_override_attr, None)
    resolved_workers = resolve_worker_count(comp.n_jobs, stage_n_jobs)
    # 1. CPU forced
    if req_backend == "cpu":
        return ComputeDecision(
            stage=stage,
            backend="cpu",
            n_jobs=resolved_workers,
            reason="CPU backend explicitly forced in compute.backend",
            device=device,
        )
    gpu_ok = is_gpu_available()
    # 2. GPU forced
    if req_backend == "gpu":
        if not gpu_ok:
            logger.warning(
                "[compute] GPU requested for stage %r but no compatible GPU was detected. Falling back to CPU.", stage
            )
            return ComputeDecision(
                stage=stage,
                backend="cpu",
                n_jobs=resolved_workers,
                reason="GPU requested but no CUDA device detected; falling back to CPU",
                device=device,
            )
        # Check if stage is GPU capable
        if stage == "clustering":
            has_rapids = is_package_available("rapids_singlecell") or is_package_available("cuml")
            if not has_rapids:
                logger.warning(
                    "[compute] GPU requested for clustering but RAPIDS (rapids-singlecell / cuml) "
                    "is not installed. Falling back to CPU Scanpy."
                )
                return ComputeDecision(
                    stage=stage,
                    backend="cpu",
                    n_jobs=resolved_workers,
                    reason="RAPIDS packages not installed; falling back to CPU",
                    device=device,
                )
            est_mem = int(n_cells * 3000 * 4 * 4)  # ~4x intermediate float32 buffer
            if not estimate_gpu_memory_safe(est_mem, device, comp.gpu_memory_fraction):
                logger.warning(
                    "[compute] GPU requested for clustering but estimated memory (%.2f GiB) exceeds "
                    "safe GPU memory. Falling back to CPU.",
                    est_mem / (1024**3),
                )
                return ComputeDecision(
                    stage=stage,
                    backend="cpu",
                    n_jobs=resolved_workers,
                    reason="Estimated clustering allocation exceeds safe GPU memory; falling back to CPU",
                    device=device,
                )
            return ComputeDecision(
                stage=stage,
                backend="gpu",
                n_jobs=1,
                reason="GPU backend explicitly requested and memory-safe",
                device=device,
            )
        elif stage == "modules_correlation":
            has_cupy = is_package_available("cupy")
            if not has_cupy:
                return ComputeDecision(
                    stage=stage,
                    backend="cpu",
                    n_jobs=resolved_workers,
                    reason="CuPy not installed; falling back to CPU",
                    device=device,
                )
            dense_elem = extra_info.get("n_dense_elements", 0)
            est_mem = int(dense_elem * 8 * 3)
            if est_mem > 0 and not estimate_gpu_memory_safe(est_mem, device, comp.gpu_memory_fraction):
                return ComputeDecision(
                    stage=stage,
                    backend="cpu",
                    n_jobs=resolved_workers,
                    reason="Estimated correlation allocation exceeds safe GPU memory; falling back to CPU",
                    device=device,
                )
            return ComputeDecision(
                stage=stage, backend="gpu", n_jobs=1, reason="GPU backend requested and memory-safe", device=device
            )
        elif stage == "distance_space":
            has_cupy = is_package_available("cupy") or is_package_available("torch")
            if not has_cupy:
                return ComputeDecision(
                    stage=stage,
                    backend="cpu",
                    n_jobs=resolved_workers,
                    reason="CuPy/PyTorch not installed for GPU distance space; using CPU",
                    device=device,
                )
            return ComputeDecision(
                stage=stage, backend="gpu", n_jobs=1, reason="GPU distance space requested", device=device
            )
        else:
            return ComputeDecision(
                stage=stage,
                backend="cpu",
                n_jobs=resolved_workers,
                reason=f"Stage {stage!r} is intentionally CPU-parallel; remaining on CPU",
                device=device,
            )
    # 3. AUTO backend
    if req_backend == "auto":
        if stage == "clustering":
            if (
                gpu_ok
                and n_cells >= comp.gpu_min_cells
                and (is_package_available("rapids_singlecell") or is_package_available("cuml"))
            ):
                est_mem = int(n_cells * 3000 * 4 * 4)
                if estimate_gpu_memory_safe(est_mem, device, comp.gpu_memory_fraction):
                    return ComputeDecision(
                        stage=stage,
                        backend="gpu",
                        n_jobs=1,
                        reason=f"AUTO: n_cells={n_cells} >= gpu_min_cells and GPU memory is safe",
                        device=device,
                    )
            return ComputeDecision(
                stage=stage,
                backend="cpu",
                n_jobs=resolved_workers,
                reason="AUTO: dataset scale or hardware suited for CPU Scanpy",
                device=device,
            )
        elif stage == "modules_correlation":
            dense_elem = extra_info.get("n_dense_elements", 0)
            if gpu_ok and dense_elem >= comp.gpu_min_dense_elements and is_package_available("cupy"):
                est_mem = int(dense_elem * 8 * 3)
                if estimate_gpu_memory_safe(est_mem, device, comp.gpu_memory_fraction):
                    return ComputeDecision(
                        stage=stage,
                        backend="gpu",
                        n_jobs=1,
                        reason=f"AUTO: dense_elements={dense_elem} >= gpu_min_dense_elements and GPU memory is safe",
                        device=device,
                    )
            return ComputeDecision(
                stage=stage,
                backend="cpu",
                n_jobs=resolved_workers,
                reason="AUTO: dense matrix size suited for CPU NumPy/SciPy",
                device=device,
            )
        elif stage == "lochness":
            return ComputeDecision(
                stage=stage,
                backend="cpu",
                n_jobs=resolved_workers,
                reason="AUTO: lochNESS executes on CPU via vectorized Numba parallel kernel",
                device=device,
            )
        elif stage in ("perturbation", "enrichment", "distance", "distance_space", "ps_score"):
            return ComputeDecision(
                stage=stage,
                backend="cpu",
                n_jobs=resolved_workers,
                reason=f"AUTO: {stage} executes on CPU with multiprocessing",
                device=device,
            )
        else:
            return ComputeDecision(
                stage=stage,
                backend="cpu",
                n_jobs=resolved_workers,
                reason=f"AUTO: {stage} executes on CPU",
                device=device,
            )
    # Fallback default
    return ComputeDecision(
        stage=stage, backend="cpu", n_jobs=resolved_workers, reason="Default CPU fallback", device=device
    )


def log_compute_decision(decision: ComputeDecision) -> None:
    """Log the compute decision formatted cleanly."""
    if decision.backend.lower() == "gpu":
        logger.info("[compute] %s backend: GPU (device %d) — %s", decision.stage, decision.device, decision.reason)
    else:
        workers_str = f" ({decision.n_jobs} workers)" if decision.n_jobs > 1 else ""
        logger.info("[compute] %s backend: CPU%s — %s", decision.stage, workers_str, decision.reason)


# Multiprocessing & Thread Safety


def derive_seed(base_seed: int, identifier: Any) -> int:
    """Derive a deterministic integer seed from a base seed and an identifier.

    Avoids Python's randomized built-in ``hash()`` to guarantee reproducible
    runs across processes and invocations.
    """
    key = f"{base_seed}_{identifier}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


@contextlib.contextmanager
def limit_blas_threads(n_threads: int = 1) -> Generator[None, None, None]:
    """Limit internal OpenMP / BLAS / MKL threads during process-parallel execution.

    Uses ``threadpoolctl`` if installed, otherwise safely falls back.
    """
    try:
        import threadpoolctl

        with threadpoolctl.threadpool_limits(limits=n_threads, user_api="blas"):
            with threadpoolctl.threadpool_limits(limits=n_threads, user_api="openmp"):
                yield
    except Exception:
        yield


def _parallel_worker_shim(func: Callable[[Any], Any], item: Any, blas_threads: int) -> Any:
    """Worker wrapper that applies threadpool limits inside child processes."""
    with limit_blas_threads(blas_threads):
        return func(item)


def run_parallel(
    func: Callable[[Any], Any], tasks: Sequence[Any], n_jobs: int = 1, blas_threads: int = 1, backend: str = "loky"
) -> List[Any]:
    """Execute target-wise tasks in parallel with thread safety and exception propagation.

    Parameters
    ----------
    func : Callable
        Worker function taking a single argument (task payload).
    tasks : Sequence[Any]
        List or sequence of task payloads.
    n_jobs : int
        Number of parallel worker processes. If <= 1, executes serially.
    blas_threads : int
        BLAS / OpenMP thread count allowed per worker process.
    backend : str
        Joblib backend engine ('loky', 'multiprocessing', 'threading').

    Returns
    -------
    List[Any]
        Results matching the exact input task order.
    """
    if len(tasks) == 0:
        return []
    if n_jobs <= 1 or len(tasks) == 1:
        with limit_blas_threads(blas_threads):
            return [func(task) for task in tasks]
    try:
        import joblib

        engine = "loky" if backend in ("process", "multiprocessing", "loky") else backend
        results = joblib.Parallel(n_jobs=n_jobs, backend=engine, return_as="list")(
            joblib.delayed(_parallel_worker_shim)(func, task, blas_threads) for task in tasks
        )
        return results
    except Exception as exc:
        logger.warning(
            "Parallel execution with backend %r encountered error (%s); falling back to serial execution.", backend, exc
        )
        with limit_blas_threads(blas_threads):
            return [func(task) for task in tasks]


# Compute Profiler & Benchmarking


@dataclass
class StageProfile:
    """Execution profile metrics for a single stage."""

    stage: str
    backend: str
    n_jobs: int
    elapsed_seconds: float
    peak_rss_mb: float
    gpu_peak_memory_mb: Optional[float] = None


class ComputeProfiler:
    """Lightweight compute profiler for tracking stage runtimes and memory."""

    def __init__(self) -> None:
        self.profiles: List[StageProfile] = []

    def record(
        self,
        stage: str,
        backend: str,
        n_jobs: int,
        elapsed_seconds: float,
        peak_rss_mb: float,
        gpu_peak_memory_mb: Optional[float] = None,
    ) -> None:
        """Record profile for a completed stage."""
        self.profiles.append(
            StageProfile(
                stage=stage,
                backend=backend.upper(),
                n_jobs=n_jobs,
                elapsed_seconds=round(elapsed_seconds, 2),
                peak_rss_mb=round(peak_rss_mb, 1),
                gpu_peak_memory_mb=(round(gpu_peak_memory_mb, 1) if gpu_peak_memory_mb is not None else None),
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return profiling records as a DataFrame."""
        if not self.profiles:
            return pd.DataFrame(
                columns=["stage", "backend", "n_jobs", "elapsed_seconds", "peak_rss_mb", "gpu_peak_memory_mb"]
            )
        return pd.DataFrame([vars(p) for p in self.profiles])

    def save_csv(self, dest: Union[str, Path]) -> Path:
        """Write profile metrics to CSV."""
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        return path


@contextlib.contextmanager
def stage_profile(
    stage: str,
    profiler: Optional[ComputeProfiler],
    decision: Optional[ComputeDecision] = None,
    backend: str = "CPU",
    n_jobs: int = 1,
    device: int = 0,
) -> Generator[None, None, None]:
    """Context manager to measure runtime and peak memory for a pipeline stage."""
    start_time = time.time()
    b_name = decision.backend if decision else backend
    w_count = decision.n_jobs if decision else n_jobs
    dev = decision.device if decision else device
    try:
        import psutil

        process = psutil.Process(os.getpid())
    except Exception:
        process = None
    yield
    elapsed = time.time() - start_time
    rss_mb = 0.0
    if process is not None:
        try:
            rss_mb = process.memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            pass
    gpu_peak_mb = None
    if b_name.lower() == "gpu":
        try:
            import torch

            if torch.cuda.is_available():
                gpu_peak_mb = torch.cuda.max_memory_allocated(dev) / (1024.0 * 1024.0)
        except Exception:
            pass
    if profiler is not None:
        profiler.record(
            stage=stage,
            backend=b_name,
            n_jobs=w_count,
            elapsed_seconds=elapsed,
            peak_rss_mb=rss_mb,
            gpu_peak_memory_mb=gpu_peak_mb,
        )
