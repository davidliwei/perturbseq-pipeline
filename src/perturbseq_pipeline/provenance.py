"""Run provenance for the basic QC stage.

Captures what is needed to reproduce a QC object: pipeline version, git branch
and commit, config path, timestamps, input files, package versions and the
SLURM/host context. Everything is plain JSON-serialisable so it can live in
``adata.uns['provenance']`` and be written next to the reports.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


_PACKAGES = (
    "scanpy",
    "anndata",
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn",
    "h5py",
    "openpyxl",
    "scikit-image",
    "scikit-learn",
)


def _git(args: Iterable[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=10)
    except Exception:  # pragma: no cover - git missing
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_info() -> Dict[str, Optional[str]]:
    """Branch / commit / dirty state of the pipeline checkout (if any)."""
    here = Path(__file__).resolve().parent
    root = _git(["rev-parse", "--show-toplevel"], here)
    if root is None:
        return {"root": None, "branch": None, "commit": None, "dirty": None}
    status = _git(["status", "--porcelain", "--untracked-files=no"], Path(root))
    return {
        "root": root,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], Path(root)),
        "commit": _git(["rev-parse", "HEAD"], Path(root)),
        "dirty": bool(status) if status is not None else None,
    }


def package_versions() -> Dict[str, str]:
    import importlib.metadata as md

    out: Dict[str, str] = {"python": platform.python_version()}
    for name in _PACKAGES:
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = "not installed"
    try:
        out["perturbseq-pipeline"] = md.version("perturbseq-pipeline")
    except Exception:
        out["perturbseq-pipeline"] = "unknown"
    return out


def slurm_info() -> Dict[str, Optional[str]]:
    keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_ACCOUNT",
    )
    return {k.lower(): os.environ.get(k) for k in keys}


def collect(
    *, config_path: Optional[str], inputs: Dict[str, Any], extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Assemble the provenance record."""
    rec: Dict[str, Any] = {
        "run_timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "user": os.environ.get("USER"),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "config_path": str(config_path) if config_path else None,
        "git": git_info(),
        "slurm": slurm_info(),
        "packages": package_versions(),
        "inputs": inputs,
    }
    if extra:
        rec.update(extra)
    return rec


def write_json(rec: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2, default=str)
    return path


def uns_safe(obj: Any) -> Any:
    """Coerce a provenance record into h5ad-writable primitives.

    ``None`` values become the string ``"None"`` because anndata cannot store
    ``NoneType`` inside ``uns``; nested dicts and lists are preserved.
    """
    if obj is None:
        return "None"
    if isinstance(obj, dict):
        return {str(k): uns_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [uns_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)
