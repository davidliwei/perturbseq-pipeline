"""Knockdown-efficiency mask.

Marks, but never removes, targeting cells whose own target gene is not knocked
down. Stage 5 still estimates perturbation strength on every cell, and both the
mask and those continuous estimates are written to the outputs, so the final
filtering decision stays with the user.

For every targeting cell ``i`` with target ``g`` in context ``c``::

    kd_ratio_i = x_ig / mean(x_g over non-targeting cells in c)

on library-size-normalized, *linear* expression (``expm1`` of the lognorm
layer). The control baseline is always per context, so in ``pooled`` mode a
context with a naturally lower baseline cannot pass for a knockdown.

Per group — (target, context) in ``per_context`` / ``any_context``, target in
``pooled`` — the filter then runs:

1. the group passes when the mean ``kd_ratio`` of its cells is below
   ``max_mean_ratio``. The baseline is constant within a context, so this is
   the ratio of mean target-cell to mean control expression. A median would
   be 0 for any target detected in under half of the cells, knockdown or not;
   the mean counts dropout zeros the same way in both groups;
2. in a passing group, cells whose own ratio is at or above ``max_cell_ratio``
   are marked as escapers;
3. a group with fewer than ``min_cells`` cells is ``non_testable`` and left
   unmarked.

``any_context`` runs step 1 per context; a target passing in at least one
context has step 2 applied in its passing contexts only and keeps every cell in
the rest. A target passing nowhere is marked everywhere it was testable.

Non-targeting, ambiguous and unassigned cells are never marked.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .cluster import LOGNORM_LAYER
from .config import Config
from .guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS, OBS_TARGET
from .perturbation import PerturbationResults

logger = logging.getLogger(__name__)

OBS_KD_RATIO = "kd_ratio"
OBS_KD_STATUS = "kd_status"
OBS_KD_KEEP = "kd_keep"

#: Context label when ``context_key`` is null.
ALL_CONTEXTS = "all"

# Per-cell statuses. The first five keep the cell, the rest mark it.
STATUS_CONTROL = "control"
STATUS_UNTOUCHED = "untouched"  # ambiguous / unassigned
STATUS_KNOCKDOWN = "knockdown"
STATUS_NON_TESTABLE = "non_testable"
STATUS_UNFILTERED_CONTEXT = "unfiltered_context"  # any_context, failing context
STATUS_ESCAPER = "escaper"
STATUS_FAILED_GROUP = "failed_group"
STATUS_LOW_EXPRESSION = "low_control_expression"
STATUS_NOT_MEASURED = "not_measured"

KEEP_STATUSES = (
    STATUS_CONTROL,
    STATUS_UNTOUCHED,
    STATUS_KNOCKDOWN,
    STATUS_NON_TESTABLE,
    STATUS_UNFILTERED_CONTEXT,
)
ALL_STATUSES = KEEP_STATUSES + (
    STATUS_ESCAPER,
    STATUS_FAILED_GROUP,
    STATUS_LOW_EXPRESSION,
    STATUS_NOT_MEASURED,
)

#: Stage-5 columns copied into the table and ``obs`` (suffixed with the control).
_STRENGTH_COLUMNS = ("log2fc", "pct_knockdown", "ks_fdr", "is_hit")


def _linear_target_columns(expr: ad.AnnData, genes: List[str]) -> sparse.csc_matrix:
    """Normalized linear expression of ``genes``, one column each.

    All targets are sliced at once: per-gene column access on a CSR matrix
    rescans every non-zero, which does not scale to a genome-wide screen.
    """
    layer = expr.layers[LOGNORM_LAYER] if LOGNORM_LAYER in expr.layers else expr.X
    sub = layer[:, [expr.var_names.get_loc(g) for g in genes]]
    sub = sparse.csc_matrix(sub, dtype=np.float64)
    sub.data = np.expm1(sub.data)
    return sub


def _contexts(expr: ad.AnnData, cfg: Config) -> np.ndarray:
    key = cfg.knockdown_filter.context_key
    if key is None:
        return np.full(expr.n_obs, ALL_CONTEXTS, dtype=object)
    if key not in expr.obs.columns:
        raise ValueError(
            f"knockdown_filter.context_key {key!r} is not an obs column; "
            f"available: {sorted(expr.obs.columns)}"
        )
    values = expr.obs[key]
    if values.isna().any():
        raise ValueError(
            f"obs[{key!r}] has {int(values.isna().sum())} missing values; every cell "
            "needs a context for the knockdown baseline."
        )
    return values.astype(str).to_numpy()


def _mark_group(
    status: np.ndarray, ratio: np.ndarray, cells: np.ndarray, max_cell_ratio: float
) -> None:
    """Step 2: split a passing group into knockdowns and escapers."""
    status[cells] = np.where(ratio[cells] < max_cell_ratio, STATUS_KNOCKDOWN, STATUS_ESCAPER)


def compute_knockdown_mask(expr: ad.AnnData, cfg: Config) -> Tuple[ad.AnnData, pd.DataFrame]:
    """Write ``kd_ratio`` / ``kd_status`` / ``kd_keep`` into ``obs``.

    Returns the AnnData (same cells, nothing removed) and a table with one row
    per (target, context).
    """
    kcfg = cfg.knockdown_filter
    obs = expr.obs
    targets = obs[OBS_TARGET].astype(str).to_numpy()
    klass = obs[OBS_CLASS].astype(str).to_numpy()
    contexts = _contexts(expr, cfg)
    context_values = sorted(set(contexts))
    mean_cut = (
        kcfg.max_mean_ratio_any
        if kcfg.mode == "any_context" and kcfg.max_mean_ratio_any is not None
        else kcfg.max_mean_ratio
    )

    ntc = klass == CLASS_NTC
    targeting = klass == CLASS_TARGETING
    status = np.full(expr.n_obs, STATUS_UNTOUCHED, dtype=object)
    status[ntc] = STATUS_CONTROL
    ratio = np.full(expr.n_obs, np.nan)

    all_targets = sorted(set(targets[targeting]))
    measured = [g for g in all_targets if g in expr.var_names]
    columns = _linear_target_columns(expr, measured) if measured else None
    col_of = {g: j for j, g in enumerate(measured)}

    rows: List[Dict[str, object]] = []
    for gene in all_targets:
        gene_cells = targeting & (targets == gene)
        x = None
        if gene in col_of:
            x = columns[:, col_of[gene]].toarray().ravel()

        # Per-context baseline and ratio; `ok` groups go on to steps 1-3.
        gene_rows: List[Dict[str, object]] = []
        for ctx in context_values:
            in_ctx = contexts == ctx
            cells = gene_cells & in_ctx
            n_cells = int(cells.sum())
            if n_cells == 0:
                continue
            ctrl = ntc & in_ctx
            row: Dict[str, object] = {
                "target_gene": gene,
                "context": ctx,
                "n_cells": n_cells,
                "n_control": int(ctrl.sum()),
                "control_mean": np.nan,
                "pct_control_expressing": np.nan,
                "mean_ratio": np.nan,
                "_cells": cells,
            }
            if x is None:
                row["group_status"] = STATUS_NOT_MEASURED
            elif row["n_control"] < kcfg.min_control_cells:
                row["group_status"] = STATUS_NON_TESTABLE
                row["reason"] = f"fewer than {kcfg.min_control_cells} control cells"
            else:
                mean = float(x[ctrl].mean())
                pct = float(100 * np.mean(x[ctrl] > 0))
                row["control_mean"] = mean
                row["pct_control_expressing"] = pct
                if pct < kcfg.min_pct_expressing_control:
                    row["group_status"] = STATUS_LOW_EXPRESSION
                else:
                    ratio[cells] = x[cells] / mean
                    row["mean_ratio"] = float(np.mean(ratio[cells]))
                    row["group_status"] = "ok"
            gene_rows.append(row)

        if kcfg.mode == "pooled":
            _decide_pooled(gene_rows, status, ratio, kcfg, mean_cut)
        else:
            _decide_per_context(gene_rows, status, ratio, kcfg, mean_cut)
        rows.extend(gene_rows)

    for row in rows:
        cells = row.pop("_cells")
        if row["group_status"] in (STATUS_NOT_MEASURED, STATUS_LOW_EXPRESSION, STATUS_NON_TESTABLE):
            status[cells] = row["group_status"]
        row["n_kept"] = int(np.isin(status[cells], KEEP_STATUSES).sum())
        row["n_escaper"] = int((status[cells] == STATUS_ESCAPER).sum())

    expr.obs[OBS_KD_RATIO] = ratio
    expr.obs[OBS_KD_STATUS] = pd.Categorical(status, categories=list(ALL_STATUSES))
    expr.obs[OBS_KD_KEEP] = np.isin(status, KEEP_STATUSES)

    table = pd.DataFrame(rows)
    if not table.empty:
        table.insert(2, "mode", kcfg.mode)
    n_marked = int((targeting & ~expr.obs[OBS_KD_KEEP].to_numpy()).sum())
    logger.info(
        "Knockdown mask (mode=%s, context=%s): %d/%d targeting cells marked for "
        "removal across %d targets; no cells removed",
        kcfg.mode,
        kcfg.context_key or "none",
        n_marked,
        int(targeting.sum()),
        len(all_targets),
    )
    return expr, table


def _decide_per_context(
    gene_rows: List[Dict[str, object]],
    status: np.ndarray,
    ratio: np.ndarray,
    kcfg,
    mean_cut: float,
) -> None:
    """``per_context`` and ``any_context``: steps 1-3 on each (target, context)."""
    testable = []
    for row in gene_rows:
        if row["group_status"] != "ok":
            continue
        if row["n_cells"] < kcfg.min_cells:
            row["group_status"] = STATUS_NON_TESTABLE
            row["reason"] = f"fewer than {kcfg.min_cells} cells"
            continue
        row["group_mean_ratio"] = row["mean_ratio"]
        row["passed_group"] = bool(row["mean_ratio"] < mean_cut)
        testable.append(row)

    any_passed = any(row["passed_group"] for row in testable)
    for row in testable:
        cells = row["_cells"]
        if row["passed_group"]:
            row["group_status"] = "pass"
            _mark_group(status, ratio, cells, kcfg.max_cell_ratio)
        elif kcfg.mode == "any_context" and any_passed:
            row["group_status"] = STATUS_UNFILTERED_CONTEXT
            status[cells] = STATUS_UNFILTERED_CONTEXT
        else:
            row["group_status"] = STATUS_FAILED_GROUP
            status[cells] = STATUS_FAILED_GROUP


def _decide_pooled(
    gene_rows: List[Dict[str, object]],
    status: np.ndarray,
    ratio: np.ndarray,
    kcfg,
    mean_cut: float,
) -> None:
    """``pooled``: one group per target over every context with a baseline.

    The mean is taken over the pooled per-cell ratios, so contexts are
    weighted by their cell count; the per-context means stay in the table.
    """
    ok = [row for row in gene_rows if row["group_status"] == "ok"]
    if not ok:
        return
    cells = np.logical_or.reduce([row["_cells"] for row in ok])
    n_cells = int(cells.sum())
    if n_cells < kcfg.min_cells:
        for row in ok:
            row["group_status"] = STATUS_NON_TESTABLE
            row["reason"] = f"fewer than {kcfg.min_cells} cells pooled"
        return
    pooled_mean = float(np.mean(ratio[cells]))
    passed = bool(pooled_mean < mean_cut)
    for row in ok:
        row["group_mean_ratio"] = pooled_mean
        row["passed_group"] = passed
        row["group_status"] = "pass" if passed else STATUS_FAILED_GROUP
    if passed:
        _mark_group(status, ratio, cells, kcfg.max_cell_ratio)
    else:
        status[cells] = STATUS_FAILED_GROUP


def attach_perturbation_strength(
    expr: ad.AnnData, table: pd.DataFrame, results: PerturbationResults
) -> Tuple[ad.AnnData, pd.DataFrame]:
    """Join stage-5 per-target estimates onto the mask table and ``obs``.

    Stage 5 runs on every cell (the mask removes none), so these are an
    independent, continuous view of the same knockdown next to the mask.
    """
    if results.table.empty or table.empty:
        return expr, table
    control = results.primary_control
    cols = [f"{c}_{control}" for c in _STRENGTH_COLUMNS if f"{c}_{control}" in results.table]
    strength = results.table.set_index("target_gene")[cols]

    table = table.merge(strength, left_on="target_gene", right_index=True, how="left")
    targets = expr.obs[OBS_TARGET].astype(str)
    for col in cols:
        values = targets.map(strength[col])
        if col.startswith("is_hit"):
            values = values.astype("boolean")
        expr.obs[f"pert_{col}"] = values.to_numpy()
    return expr, table
