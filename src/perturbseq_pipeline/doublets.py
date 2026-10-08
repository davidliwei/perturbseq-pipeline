"""Doublet detection for the basic QC stage — annotation only.

Scrublet (the scanpy implementation, ``sc.pp.scrublet``) is run independently
per GEM well. Results are written to ``obs`` as ``doublet_score`` and
``predicted_doublet``; the sample's threshold and settings go to
``uns['scrublet']``. This module never removes or subsets cells, and it does
not force an expected doublet rate onto the call.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .config import Config

logger = logging.getLogger(__name__)

DOUBLET_SCORE = "doublet_score"
PREDICTED_DOUBLET = "predicted_doublet"


def assess_threshold(scores: np.ndarray, threshold: Optional[float], n_predicted: int, mode: str) -> Dict[str, object]:
    """Describe the score distribution and judge whether an automatic threshold is plausible.

    ``threshold_suspect`` is True when an *automatic* threshold sits above the
    99.9th percentile of observed scores or calls fewer than 0.5% of cells:
    Scrublet's bimodality search then likely failed (near-homogeneous
    populations), and the calls should not be trusted as-is. Nothing is forced;
    the scores are kept so a threshold can be chosen after inspection
    (``qc.doublets.threshold``).
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    out: Dict[str, object] = {}
    if s.size:
        for q in (50, 90, 99, 99.9):
            out[f"score_p{q}".replace(".", "_")] = float(np.percentile(s, q))
        out["score_max"] = float(s.max())
    frac = n_predicted / max(len(scores), 1)
    suspect = False
    reason = ""
    if mode == "auto" and threshold is not None and s.size:
        if threshold > out["score_p99_9"]:
            suspect, reason = True, "threshold above the 99.9th percentile of observed scores"
        elif frac < 0.005:
            suspect, reason = True, "fewer than 0.5% of cells called"
    out["threshold_suspect"] = suspect
    out["threshold_suspect_reason"] = reason
    return out


def run_scrublet(expr: ad.AnnData, cfg: Config, sample_id: str, seed: Optional[int] = None) -> Dict[str, object]:
    """Annotate ``expr`` with Scrublet scores and calls. Returns a summary dict.

    ``expr.X`` must hold raw counts. The matrix is passed to Scrublet as a
    lightweight copy (no layers) so peak memory is roughly 2x the count matrix
    rather than 3x; ``expr`` itself is only touched in ``obs``/``uns``.
    """
    dcfg = cfg.qc.doublets
    n_before = expr.n_obs
    summary: Dict[str, object] = {
        "sample_id": sample_id,
        "method": dcfg.method,
        "enabled": dcfg.enabled,
        "n_cells": n_before,
        "expected_doublet_rate": dcfg.expected_doublet_rate if dcfg.expected_doublet_rate is not None else 0.05,
        "threshold": None,
        "threshold_mode": "manual" if dcfg.threshold is not None else "auto",
        "threshold_failed": False,
        "n_predicted_doublets": 0,
        "fraction_predicted_doublets": float("nan"),
        "n_scored": 0,
        "elapsed_seconds": 0.0,
    }
    if not dcfg.enabled:
        expr.obs[DOUBLET_SCORE] = np.nan
        expr.obs[PREDICTED_DOUBLET] = False
        summary["threshold_mode"] = "disabled"
        return summary
    t0 = time.time()
    light = ad.AnnData(
        X=expr.X, obs=pd.DataFrame(index=expr.obs_names.copy()), var=pd.DataFrame(index=expr.var_names.copy())
    )
    kwargs = dict(
        sim_doublet_ratio=dcfg.sim_doublet_ratio,
        expected_doublet_rate=summary["expected_doublet_rate"],
        stdev_doublet_rate=dcfg.stdev_doublet_rate,
        n_prin_comps=dcfg.n_prin_comps,
        threshold=dcfg.threshold,
        random_state=int(seed if seed is not None else cfg.run.seed),
        verbose=False,
        copy=False,
    )
    try:
        sc.pp.scrublet(light, **kwargs)
    except Exception as exc:  # pragma: no cover - depends on data
        logger.warning("%s: Scrublet failed (%s); recording NaN scores and no calls", sample_id, exc)
        expr.obs[DOUBLET_SCORE] = np.nan
        expr.obs[PREDICTED_DOUBLET] = False
        summary["threshold_failed"] = True
        summary["error"] = str(exc)
        summary["elapsed_seconds"] = time.time() - t0
        return summary
    score = pd.to_numeric(light.obs[DOUBLET_SCORE], errors="coerce").reindex(expr.obs_names)
    pred = light.obs[PREDICTED_DOUBLET].reindex(expr.obs_names)
    scr_uns = light.uns.get("scrublet", {}) if hasattr(light, "uns") else {}
    threshold = scr_uns.get("threshold", None)
    threshold_ok = threshold is not None and np.isfinite(float(threshold))
    if not threshold_ok or pred.isna().all():
        logger.warning(
            "%s: Scrublet could not set an automatic threshold; storing scores only "
            "(predicted_doublet=False for every cell)",
            sample_id,
        )
        summary["threshold_failed"] = True
        pred_bool = np.zeros(n_before, dtype=bool)
    else:
        pred_bool = pred.fillna(False).astype(bool).to_numpy()
        summary["threshold"] = float(threshold)
    expr.obs[DOUBLET_SCORE] = score.to_numpy(dtype=float)
    expr.obs[PREDICTED_DOUBLET] = pred_bool
    expr.uns["scrublet"] = {
        "threshold": float(threshold) if threshold_ok else "None",
        "expected_doublet_rate": float(summary["expected_doublet_rate"]),
        "sim_doublet_ratio": float(dcfg.sim_doublet_ratio),
        "n_prin_comps": int(dcfg.n_prin_comps),
        "random_state": int(kwargs["random_state"]),
        "threshold_mode": summary["threshold_mode"],
        "threshold_failed": bool(summary["threshold_failed"]),
    }
    for key in ("doublet_rate_prediction", "detectable_doublet_fraction", "overall_doublet_rate"):
        if key in scr_uns:
            try:
                expr.uns["scrublet"][key] = float(scr_uns[key])
                summary[key] = float(scr_uns[key])
            except Exception:
                pass
    summary["n_scored"] = int(np.isfinite(expr.obs[DOUBLET_SCORE].to_numpy(dtype=float)).sum())
    summary["n_predicted_doublets"] = int(pred_bool.sum())
    summary["fraction_predicted_doublets"] = float(pred_bool.mean()) if n_before else float("nan")
    diag = assess_threshold(
        expr.obs[DOUBLET_SCORE].to_numpy(dtype=float),
        summary["threshold"],
        summary["n_predicted_doublets"],
        summary["threshold_mode"],
    )
    summary.update(diag)
    expr.uns["scrublet"]["threshold_suspect"] = bool(diag["threshold_suspect"])
    expr.uns["scrublet"]["threshold_suspect_reason"] = diag["threshold_suspect_reason"] or "None"
    if diag["threshold_suspect"]:
        logger.warning(
            "%s: Scrublet automatic threshold %.3f looks implausible (%s); calls kept as annotation only — "
            "inspect doublet_score and consider qc.doublets.threshold",
            sample_id,
            summary["threshold"],
            diag["threshold_suspect_reason"],
        )
    summary["elapsed_seconds"] = time.time() - t0
    assert expr.n_obs == n_before, "Scrublet wrapper must never change the number of cells"
    logger.info(
        "%s: Scrublet %d/%d cells flagged as predicted doublets (threshold=%s, %.0fs); cells retained: %d",
        sample_id,
        summary["n_predicted_doublets"],
        n_before,
        f"{summary['threshold']:.3f}" if summary["threshold"] is not None else "n/a",
        summary["elapsed_seconds"],
        expr.n_obs,
    )
    return summary
