"""Figures for the basic QC stage.

All plots work from ``obs`` tables so they scale to a few hundred thousand
cells without touching the count matrix. Flagged cells (predicted doublets,
guide multiplets) are *shown*, never dropped: they are overlaid as a second
series on every distribution where the distinction matters.

Colour assignment is fixed by meaning (never cycled):
    all cells / reference  -> blue
    predicted doublet      -> orange
    guide multiplet        -> violet
    guide clean            -> aqua
Scaffold classes take the categorical slots in configured order.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import FigureRegistry, SECTION_QC, SECTION_GUIDES

logger = logging.getLogger(__name__)

SECTION_DOUBLETS = "doublets"
SECTION_QC_SAMPLES = "qc/per_sample"

C_ALL = "#2a78d6"
C_DOUBLET = "#eb6834"
C_MULTIPLET = "#4a3aa7"
C_CLEAN = "#1baf7a"
C_FAIL = "#e34948"
C_GRID = "#d9d8d4"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

MAX_SCATTER_POINTS = 60_000


def _style(ax):
    ax.grid(True, color=C_GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _subsample(n: int, seed: int = 0, max_points: int = MAX_SCATTER_POINTS) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def _log_hist(ax, values: np.ndarray, color: str, label: str, bins: np.ndarray, alpha: float = 0.85):
    ax.hist(
        values, bins=bins, color=color, alpha=alpha, label=label, histtype="stepfilled", linewidth=0.8, edgecolor=color
    )


# Expression QC per sample


def plot_sample_expression_qc(
    obs: pd.DataFrame, thresholds: Dict[str, object], sample_id: str, reg: FigureRegistry
) -> None:
    """Histograms of the four core metrics with the resolved thresholds drawn."""
    metrics = [
        ("total_counts", "Total UMI counts", True, ("min_counts", "max_counts")),
        ("n_genes_by_counts", "Genes detected", True, ("min_genes", "max_genes")),
        ("pct_counts_mt", "Mitochondrial %", False, (None, "max_pct_mt")),
        ("pct_counts_ribo", "Ribosomal %", False, (None, None)),
    ]
    metrics = [m for m in metrics if m[0] in obs.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
    axes = np.atleast_1d(axes)
    dbl = obs["predicted_doublet"].astype(bool).to_numpy() if "predicted_doublet" in obs else None
    for ax, (col, label, logx, (lo_key, hi_key)) in zip(axes, metrics):
        v = obs[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if logx:
            pos = v[v > 0]
            bins = np.logspace(np.log10(max(pos.min(), 1)), np.log10(pos.max() + 1), 60) if pos.size else 30
            ax.set_xscale("log")
        else:
            bins = np.linspace(0, max(np.nanpercentile(v, 99.9) * 1.05, 1e-3), 60) if v.size else 30
        _log_hist(ax, v, C_ALL, "all cells", bins)
        if dbl is not None and dbl.any():
            _log_hist(ax, obs.loc[dbl, col].to_numpy(dtype=float), C_DOUBLET, "predicted doublet", bins, alpha=0.9)
        for key, style in ((lo_key, "--"), (hi_key, "--")):
            thr = thresholds.get(key) if key else None
            if thr is not None and np.isfinite(float(thr)):
                ax.axvline(float(thr), color=C_FAIL, linestyle=style, linewidth=1.2, label=f"{key} = {float(thr):,.0f}")
        ax.set_xlabel(label)
        ax.set_ylabel("cells")
        _style(ax)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{sample_id}: expression QC metrics (n = {len(obs):,} cells; nothing removed)", fontsize=10)
    fig.tight_layout()
    reg.save(
        fig,
        f"{sample_id}_qc_metrics",
        SECTION_QC_SAMPLES,
        f"{sample_id} — QC metric distributions",
        "Dashed lines show the resolved per-sample thresholds; predicted doublets are overlaid, not removed.",
    )
    # counts vs genes / counts vs mt scatter
    idx = _subsample(len(obs))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    x = obs["total_counts"].to_numpy(dtype=float)[idx]
    for ax, ycol, ylabel in zip(axes, ("n_genes_by_counts", "pct_counts_mt"), ("Genes detected", "Mitochondrial %")):
        if ycol not in obs:
            continue
        y = obs[ycol].to_numpy(dtype=float)[idx]
        sub_dbl = dbl[idx] if dbl is not None else np.zeros(len(idx), dtype=bool)
        ax.scatter(
            x[~sub_dbl],
            y[~sub_dbl],
            s=3,
            alpha=0.35,
            color=C_ALL,
            linewidths=0,
            label="singlet / unscored",
            rasterized=True,
        )
        if sub_dbl.any():
            ax.scatter(
                x[sub_dbl],
                y[sub_dbl],
                s=4,
                alpha=0.6,
                color=C_DOUBLET,
                linewidths=0,
                label="predicted doublet",
                rasterized=True,
            )
        ax.set_xscale("log")
        if ycol == "n_genes_by_counts":
            ax.set_yscale("log")
        for key, axis in (("min_counts", "x"), ("max_counts", "x")):
            thr = thresholds.get(key)
            if thr is not None and np.isfinite(float(thr)):
                ax.axvline(float(thr), color=C_FAIL, linestyle="--", linewidth=1)
        hk = "min_genes" if ycol == "n_genes_by_counts" else "max_pct_mt"
        thr = thresholds.get(hk)
        if thr is not None and np.isfinite(float(thr)):
            ax.axhline(float(thr), color=C_FAIL, linestyle="--", linewidth=1)
        if ycol == "n_genes_by_counts":
            thr = thresholds.get("max_genes")
            if thr is not None and np.isfinite(float(thr)):
                ax.axhline(float(thr), color=C_FAIL, linestyle="--", linewidth=1)
        ax.set_xlabel("Total UMI counts")
        ax.set_ylabel(ylabel)
        _style(ax)
        ax.legend(fontsize=7, frameon=False, markerscale=3)
    fig.suptitle(f"{sample_id}: counts vs genes / counts vs mt ({len(idx):,} of {len(obs):,} cells shown)", fontsize=10)
    fig.tight_layout()
    reg.save(
        fig,
        f"{sample_id}_qc_scatter",
        SECTION_QC_SAMPLES,
        f"{sample_id} — counts vs genes and mt%",
        "Dashed lines: resolved thresholds. Orange: Scrublet predicted doublets (retained).",
    )


# Doublets


def plot_doublet_scores(obs: pd.DataFrame, sample_id: str, threshold: Optional[float], reg: FigureRegistry) -> None:
    if "doublet_score" not in obs:
        return
    score = pd.to_numeric(obs["doublet_score"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(score)
    if not ok.any():
        return
    has_guide = "guide_multiplet_flag" in obs
    fig, axes = plt.subplots(1, 2 if has_guide else 1, figsize=(9 if has_guide else 4.8, 3.6))
    axes = np.atleast_1d(axes)
    bins = np.linspace(0, max(score[ok].max(), 0.05), 60)
    ax = axes[0]
    _log_hist(ax, score[ok], C_ALL, "all scored cells", bins)
    if threshold is not None and np.isfinite(threshold):
        ax.axvline(threshold, color=C_FAIL, linestyle="--", linewidth=1.2, label=f"threshold = {threshold:.3f}")
    ax.set_xlabel("Scrublet doublet score")
    ax.set_ylabel("cells")
    ax.set_yscale("log")
    _style(ax)
    ax.legend(fontsize=7, frameon=False)
    if has_guide:
        ax = axes[1]
        gm = obs["guide_multiplet_flag"].astype(bool).to_numpy()
        det = obs["guide_detected"].astype(bool).to_numpy() if "guide_detected" in obs else np.ones(len(obs), bool)
        _log_hist(ax, score[ok & ~gm & det], C_CLEAN, "guide clean", bins, alpha=0.7)
        _log_hist(ax, score[ok & gm], C_MULTIPLET, "guide multiplet", bins, alpha=0.7)
        if threshold is not None and np.isfinite(threshold):
            ax.axvline(threshold, color=C_FAIL, linestyle="--", linewidth=1.2)
        ax.set_xlabel("Scrublet doublet score")
        ax.set_ylabel("cells")
        ax.set_yscale("log")
        _style(ax)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{sample_id}: Scrublet scores (annotation only; no cells removed)", fontsize=10)
    fig.tight_layout()
    reg.save(
        fig,
        f"{sample_id}_doublet_scores",
        SECTION_DOUBLETS,
        f"{sample_id} — Scrublet doublet scores",
        "Left: all cells with the automatic threshold. Right: split by the guide-derived multiplet flag.",
    )


def plot_scrublet_vs_guide(obs: pd.DataFrame, label: str, reg: FigureRegistry) -> None:
    if "predicted_doublet" not in obs or "guide_multiplet_flag" not in obs:
        return
    pred = obs["predicted_doublet"].astype(bool).to_numpy()
    gm = obs["guide_multiplet_flag"].astype(bool).to_numpy()
    det = obs["guide_detected"].astype(bool).to_numpy() if "guide_detected" in obs else np.ones(len(obs), bool)
    table = np.array(
        [
            [int((~pred & ~gm & det).sum()), int((~pred & gm).sum()), int((~pred & ~det).sum())],
            [int((pred & ~gm & det).sum()), int((pred & gm).sum()), int((pred & ~det).sum())],
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    im = ax.imshow(table, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["guide clean", "guide multiplet", "no guide"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Scrublet singlet", "Scrublet doublet"])
    total = table.sum() or 1
    for i in range(2):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{table[i, j]:,}\n({100 * table[i, j] / total:.1f}%)",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if table[i, j] > table.max() / 2 else "#0b0b0b",
            )
    ax.set_title("Scrublet call vs guide multiplet flag", fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax = axes[1]
    if "doublet_score" in obs:
        score = pd.to_numeric(obs["doublet_score"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(score)
        groups = [
            ("guide clean", score[ok & ~gm & det], C_CLEAN),
            ("guide multiplet", score[ok & gm], C_MULTIPLET),
            ("no guide", score[ok & ~det], "#52514e"),
        ]
        groups = [g for g in groups if g[1].size]
    else:
        groups = []
    if groups:
        parts = ax.boxplot(
            [g[1] for g in groups],
            tick_labels=[f"{g[0]}\n(n={g[1].size:,})" for g in groups],
            showfliers=False,
            patch_artist=True,
            widths=0.55,
        )
        for patch, g in zip(parts["boxes"], groups):
            patch.set_facecolor(g[2])
            patch.set_alpha(0.75)
        ax.set_ylabel("Scrublet doublet score")
        _style(ax)
    else:
        ax.text(0.5, 0.5, "no Scrublet scores", ha="center", va="center", transform=ax.transAxes, color="#52514e")
        ax.set_axis_off()
    fig.suptitle(f"{label}: Scrublet vs guide-derived multiplets (assessment only)", fontsize=10)
    fig.tight_layout()
    reg.save(
        fig,
        f"{label}_scrublet_vs_guide_multiplet",
        SECTION_DOUBLETS,
        f"{label} — Scrublet vs guide multiplet",
        "Both indicators are stored as flags; no cell was removed.",
    )


# Guides


def plot_guide_qc(obs: pd.DataFrame, sample_id: str, classes: Sequence[str], reg: FigureRegistry) -> None:
    if "guide_umi_total" not in obs:
        return
    n_panels = 2 + len(classes)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.6))
    axes = np.atleast_1d(axes)
    dbl = obs["predicted_doublet"].astype(bool).to_numpy() if "predicted_doublet" in obs else np.zeros(len(obs), bool)
    ax = axes[0]
    v = obs["guide_umi_total"].to_numpy(dtype=float)
    pos = v[v > 0]
    if pos.size:
        bins = np.logspace(0, np.log10(pos.max() + 1), 60)
        _log_hist(ax, pos, C_ALL, "all cells", bins)
        if dbl.any():
            _log_hist(ax, v[dbl & (v > 0)], C_DOUBLET, "predicted doublet", bins)
        ax.set_xscale("log")
    ax.set_xlabel("Guide UMIs per cell (cells with > 0)")
    ax.set_ylabel("cells")
    ax.set_title(f"{int((v == 0).sum()):,} cells with 0 guide UMIs", fontsize=8)
    _style(ax)
    ax.legend(fontsize=7, frameon=False)

    def _bar(ax, col, title):
        vals = obs[col].to_numpy()
        cap = 6
        capped = np.minimum(vals, cap)
        labels = [str(i) for i in range(cap)] + [f"≥{cap}"]
        tot = np.bincount(capped, minlength=cap + 1)
        dbl_c = np.bincount(capped[dbl], minlength=cap + 1) if dbl.any() else np.zeros(cap + 1, int)
        x = np.arange(cap + 1)
        ax.bar(x, tot - dbl_c, color=C_ALL, width=0.7, label="singlet / unscored")
        ax.bar(x, dbl_c, bottom=tot - dbl_c, color=C_DOUBLET, width=0.7, label="predicted doublet")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(title)
        ax.set_ylabel("cells")
        _style(ax)
        ax.legend(fontsize=7, frameon=False)

    _bar(axes[1], "n_guides", "Detected guides per cell")
    for ax, c in zip(axes[2:], classes):
        col = f"n_guides_{c}"
        if col in obs:
            _bar(ax, col, f"Scaffold {c} guides per cell")
    fig.suptitle(f"{sample_id}: guide QC (n = {len(obs):,} cells)", fontsize=10)
    fig.tight_layout()
    reg.save(
        fig,
        f"{sample_id}_guide_qc",
        SECTION_GUIDES,
        f"{sample_id} — guide UMIs and guides per cell",
        "Detected = UMIs at or above the detection threshold. Predicted doublets are stacked, not removed.",
    )
    if "total_counts" in obs and (v > 0).any():
        idx = _subsample(len(obs))
        fig, ax = plt.subplots(figsize=(4.8, 4))
        gm = (
            obs["guide_multiplet_flag"].astype(bool).to_numpy()[idx]
            if "guide_multiplet_flag" in obs
            else np.zeros(len(idx), bool)
        )
        x = obs["total_counts"].to_numpy(dtype=float)[idx]
        y = v[idx]
        ax.scatter(
            x[~gm],
            y[~gm] + 1,
            s=3,
            alpha=0.35,
            color=C_CLEAN,
            linewidths=0,
            label="guide clean / none",
            rasterized=True,
        )
        if gm.any():
            ax.scatter(
                x[gm],
                y[gm] + 1,
                s=4,
                alpha=0.6,
                color=C_MULTIPLET,
                linewidths=0,
                label="guide multiplet",
                rasterized=True,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Total GEX UMI counts")
        ax.set_ylabel("Guide UMIs per cell + 1")
        _style(ax)
        ax.legend(fontsize=7, frameon=False, markerscale=3)
        fig.suptitle(f"{sample_id}: guide UMIs vs GEX UMIs", fontsize=10)
        fig.tight_layout()
        reg.save(
            fig,
            f"{sample_id}_guide_vs_gex_umis",
            SECTION_GUIDES,
            f"{sample_id} — guide vs GEX UMIs",
            "Guide multiplet flags shown in violet; retained in every output.",
        )


def plot_guide_representation(design: pd.DataFrame, umis_by_sample: Dict[str, np.ndarray], reg: FigureRegistry) -> None:
    """Guide UMI representation across samples (log scale) with zero-count guides marked."""
    if not umis_by_sample:
        return
    fig, ax = plt.subplots(figsize=(7, 3.8))
    n = len(design)
    for i, (sid, umis) in enumerate(umis_by_sample.items()):
        order = np.argsort(-umis)
        ax.plot(
            np.arange(n),
            np.sort(umis)[::-1] + 1,
            color=CATEGORICAL[i % len(CATEGORICAL)],
            linewidth=1.6,
            label=f"{sid} ({int((umis > 0).sum())}/{n} observed)",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Designed guides, ranked by UMI count")
    ax.set_ylabel("Guide UMIs in cells + 1")
    _style(ax)
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Guide representation per sample (all designed guides)", fontsize=10)
    fig.tight_layout()
    reg.save(
        fig,
        "guide_representation",
        SECTION_GUIDES,
        "Guide representation",
        "Every designed guide is a column of the count matrix; zero-count guides sit at 1 on this axis.",
    )


# Combined


def plot_combined_metrics(obs: pd.DataFrame, reg: FigureRegistry, sample_key: str = "sample_id") -> None:
    metrics = [m for m in ("total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo") if m in obs]
    samples = list(pd.unique(obs[sample_key].astype(str)))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.8))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        data = [obs.loc[obs[sample_key].astype(str) == s, m].to_numpy(dtype=float) for s in samples]
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for body, i in zip(parts["bodies"], range(len(samples))):
            body.set_facecolor(CATEGORICAL[i % len(CATEGORICAL)])
            body.set_alpha(0.75)
        ax.set_xticks(np.arange(1, len(samples) + 1))
        ax.set_xticklabels(samples, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(m)
        if m in ("total_counts", "n_genes_by_counts"):
            ax.set_yscale("log")
        _style(ax)
    fig.suptitle(f"Combined QC metrics by sample (n = {len(obs):,} cells, all retained)", fontsize=10)
    fig.tight_layout()
    reg.save(
        fig,
        "combined_qc_metrics_by_sample",
        SECTION_QC,
        "Combined QC metrics by sample",
        "Per-sample distributions on the all-cells object; medians marked.",
    )


def plot_sample_flag_summary(summary: pd.DataFrame, reg: FigureRegistry) -> None:
    """Grouped bars: fraction of cells per sample carrying each flag."""
    cols = [
        c
        for c in ("frac_gex_qc_pass", "frac_predicted_doublet", "frac_guide_detected", "frac_guide_multiplet_flag")
        if c in summary
    ]
    if not cols or summary.empty:
        return
    labels = {
        "frac_gex_qc_pass": "expression QC pass",
        "frac_predicted_doublet": "Scrublet flagged",
        "frac_guide_detected": "guide detected",
        "frac_guide_multiplet_flag": "guide multiplet flagged",
    }
    colors = {
        "frac_gex_qc_pass": C_ALL,
        "frac_predicted_doublet": C_DOUBLET,
        "frac_guide_detected": C_CLEAN,
        "frac_guide_multiplet_flag": C_MULTIPLET,
    }
    fig, ax = plt.subplots(figsize=(1.6 * len(summary) + 3, 3.6))
    x = np.arange(len(summary))
    w = 0.8 / len(cols)
    for i, c in enumerate(cols):
        ax.bar(
            x + (i - (len(cols) - 1) / 2) * w,
            summary[c].to_numpy(dtype=float),
            width=w * 0.92,
            color=colors[c],
            label=labels[c],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["sample_id"].astype(str), fontsize=8)
    ax.set_ylabel("fraction of input cells")
    ax.set_ylim(0, 1.05)
    _style(ax)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("Per-sample QC flag fractions (flags only — no cells removed)", fontsize=10)
    fig.tight_layout()
    reg.save(
        fig,
        "sample_flag_summary",
        SECTION_QC,
        "Per-sample flag summary",
        "Scrublet flagged and guide multiplet flagged are annotations, not removals.",
    )


def plot_guide_detection_sensitivity(table: pd.DataFrame, reg: FigureRegistry) -> None:
    """structure_pass and multiplet fractions across detection rules, per sample."""
    if table is None or table.empty:
        return
    samples = list(pd.unique(table["sample_id"].astype(str)))
    fracs = sorted(table["min_fraction_of_top"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, col, ylabel in zip(
        axes,
        ("frac_guide_structure_pass", "frac_guide_multiplet_flag"),
        ("fraction structure_pass", "fraction guide_multiplet_flag"),
    ):
        for i, frac in enumerate(fracs):
            sub = table[table["min_fraction_of_top"] == frac]
            mean = sub.groupby("detection_threshold_umi")[col].mean()
            ax.plot(
                mean.index,
                mean.to_numpy(),
                marker="o",
                markersize=4,
                linewidth=1.6,
                color=CATEGORICAL[i % len(CATEGORICAL)],
                label=f"min fraction of top = {frac:g}",
            )
        cur = table[table["is_current_rule"]]
        if len(cur):
            ax.scatter(
                cur["detection_threshold_umi"],
                cur[col],
                s=70,
                facecolors="none",
                edgecolors=C_FAIL,
                linewidths=1.5,
                label="configured rule",
                zorder=5,
            )
        ax.set_xscale("log")
        ax.set_xlabel("absolute detection threshold (UMIs)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.02)
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Guide detection-rule sensitivity (mean over {len(samples)} sample(s))", fontsize=10)
    fig.tight_layout()
    reg.save(
        fig,
        "guide_detection_sensitivity",
        SECTION_GUIDES,
        "Guide detection-rule sensitivity",
        "How the structure-pass and multiplet fractions move with the detection rule. Assessment only; "
        "the stored flags use the configured rule (red circle).",
    )


def plot_doublet_score_by_sample(
    obs: pd.DataFrame, doublet_rows: Sequence[Dict[str, object]], reg: FigureRegistry, sample_key: str = "sample_id"
) -> None:
    """Scrublet score distributions per sample with each automatic threshold."""
    if "doublet_score" not in obs:
        return
    samples = list(pd.unique(obs[sample_key].astype(str)))
    thr_of = {str(r.get("sample_id")): r.get("threshold") for r in doublet_rows}
    suspect = {str(r.get("sample_id")): bool(r.get("threshold_suspect", False)) for r in doublet_rows}
    fig, axes = plt.subplots(1, len(samples), figsize=(3.6 * len(samples), 3.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, samples):
        score = pd.to_numeric(obs.loc[obs[sample_key].astype(str) == s, "doublet_score"], errors="coerce").to_numpy(
            dtype=float
        )
        score = score[np.isfinite(score)]
        if not score.size:
            ax.set_axis_off()
            continue
        bins = np.linspace(0, max(score.max(), 0.05), 50)
        _log_hist(ax, score, C_ALL, "cells", bins)
        thr = thr_of.get(s)
        if thr is not None and np.isfinite(float(thr)):
            ax.axvline(
                float(thr),
                color=C_FAIL,
                linestyle="--",
                linewidth=1.2,
                label=f"auto threshold {float(thr):.2f}" + (" (suspect)" if suspect.get(s) else ""),
            )
        ax.set_yscale("log")
        ax.set_title(s, fontsize=9)
        ax.set_xlabel("Scrublet doublet score")
        _style(ax)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Scrublet score distributions per sample (annotation only)", fontsize=10)
    fig.tight_layout()
    reg.save(
        fig,
        "doublet_scores_by_sample",
        SECTION_DOUBLETS,
        "Scrublet scores by sample",
        "Automatic thresholds marked; 'suspect' means the threshold sits beyond the observed score range or calls <0.5% of cells.",
    )
