"""Knockdown mask: every mode on a hand-built dataset with known answers.

Values are constructed directly in normalized units, so each cell's ratio to
its context's control mean — and therefore its status — is known exactly.

    target  context  cells  expression        control mean
    G1      A        50     1   (ratio 0.1)   10
    G1      A        10     10  (ratio 1.0)   10
    G1      B        30     10  (ratio 1.0)   10
    G2      A        5      1                 10     too few cells
    G3      A        40     0                 ~0     barely expressed in controls
    G4      A        35     -                 -      not in the expression matrix
    G5      B        40     2   (ratio 1.0)   2 in B, 20 in A
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))

from perturbseq_pipeline.cluster import LOGNORM_LAYER  # noqa: E402
from perturbseq_pipeline.config import Config  # noqa: E402
from perturbseq_pipeline.knockdown_filter import (  # noqa: E402
    OBS_KD_KEEP,
    OBS_KD_RATIO,
    OBS_KD_STATUS,
    compute_knockdown_mask,
)

GENES = ["G1", "G2", "G3", "G5", "OTHER"]


def _block(target, klass, ctx, n, **expression):
    return [(target, klass, ctx, {g: expression.get(g, 5.0) for g in GENES})] * n


@pytest.fixture
def adata():
    rows = []
    for ctx, g5 in (("A", 20.0), ("B", 2.0)):
        ctrl = _block("non-targeting", "non-targeting", ctx, 40, G1=10, G2=10, G3=0, G5=g5)
        rows += ctrl
    # One control cell in A expresses G3: 1/40 = 2.5% detection, below 10%.
    t, k, c, e = rows[0]
    rows[0] = (t, k, c, {**e, "G3": 4.0})
    rows += _block("G1", "targeting", "A", 50, G1=1)
    rows += _block("G1", "targeting", "A", 10, G1=10)
    rows += _block("G1", "targeting", "B", 30, G1=10)
    rows += _block("G2", "targeting", "A", 5, G2=1)
    rows += _block("G3", "targeting", "A", 40, G3=0)
    rows += _block("G4", "targeting", "A", 35)
    rows += _block("G5", "targeting", "B", 40, G5=2)
    rows += _block("ambiguous", "ambiguous", "A", 10, G1=10)
    values = np.array([[e[g] for g in GENES] for _, _, _, e in rows])
    obs = pd.DataFrame(
        {
            "target_gene": [r[0] for r in rows],
            "perturbation_class": [r[1] for r in rows],
            "cell_line": [r[2] for r in rows],
        },
        index=[f"cell{i}" for i in range(len(rows))],
    )
    a = ad.AnnData(X=sp.csr_matrix(values), obs=obs, var=pd.DataFrame(index=GENES))
    a.layers[LOGNORM_LAYER] = sp.csr_matrix(np.log1p(values))
    return a


def _run(adata, **kd):
    cfg = Config.from_dict({"knockdown_filter": {"enabled": True, **kd}})
    return compute_knockdown_mask(adata, cfg)


def _statuses(a, target, ctx=None):
    obs = a.obs
    m = obs["target_gene"] == target
    if ctx is not None:
        m &= obs["cell_line"] == ctx
    return obs.loc[m, OBS_KD_STATUS].astype(str).value_counts().to_dict()


def test_per_context_filters_each_context_independently(adata):
    a, table = _run(adata, mode="per_context", context_key="cell_line")
    assert a.n_obs == adata.n_obs  # a mask, never a filter
    assert _statuses(a, "G1", "A") == {"knockdown": 50, "escaper": 10}
    assert _statuses(a, "G1", "B") == {"failed_group": 30}
    assert _statuses(a, "G2") == {"non_testable": 5}
    assert _statuses(a, "G3") == {"low_control_expression": 40}
    assert _statuses(a, "G4") == {"not_measured": 35}
    assert _statuses(a, "G5") == {"failed_group": 40}
    row = table.set_index(["target_gene", "context"]).loc[("G1", "A")]
    assert row["mean_ratio"] == pytest.approx((50 * 0.1 + 10 * 1.0) / 60)
    assert row["n_kept"] == 50 and row["n_escaper"] == 10


def test_any_context_keeps_every_cell_in_failing_contexts(adata):
    a, _ = _run(adata, mode="any_context", context_key="cell_line")
    assert _statuses(a, "G1", "A") == {"knockdown": 50, "escaper": 10}
    assert _statuses(a, "G1", "B") == {"unfiltered_context": 30}
    assert a.obs.loc[(a.obs["target_gene"] == "G1") & (a.obs["cell_line"] == "B"), OBS_KD_KEEP].all()
    # G5 passes nowhere, so it is marked like in per_context.
    assert _statuses(a, "G5") == {"failed_group": 40}


def test_any_context_uses_the_stricter_threshold(adata):
    a, _ = _run(adata, mode="any_context", context_key="cell_line", max_mean_ratio_any=0.05)
    assert _statuses(a, "G1") == {"failed_group": 90}


def test_pooled_mean_uses_per_context_baselines(adata):
    a, table = _run(adata, mode="pooled", context_key="cell_line")
    # Pooled over A and B, G1's mean ratio is (50 * 0.1 + 40 * 1.0) / 90 = 0.5:
    # context B, where G1 is not knocked down at all, pulls the target over the cut.
    assert _statuses(a, "G1") == {"failed_group": 90}
    assert np.allclose(table.loc[table["target_gene"] == "G1", "group_mean_ratio"], 0.5)
    # G5 is at its own context's baseline: not a knockdown.
    assert _statuses(a, "G5") == {"failed_group": 40}


def test_pooled_without_context_is_fooled_by_baseline_differences(adata):
    """The motivating case for per-context baselines: G5 only looks knocked
    down against a baseline averaged over a high-expressing context."""
    a, _ = _run(adata, mode="pooled", context_key=None)
    assert _statuses(a, "G5") == {"knockdown": 40}
    assert a.obs.loc[a.obs["target_gene"] == "G5", OBS_KD_RATIO].iloc[0] == pytest.approx(2 / 11)


def test_dropout_alone_does_not_pass_as_knockdown():
    """A target detected in 30% of controls, and not knocked down: 70% of its
    cells are zero, so the median ratio would be 0 and pass. The mean is 1."""
    detected = [3.0] * 12 + [0.0] * 28
    values = np.array([[v] for v in detected * 2])
    obs = pd.DataFrame(
        {
            "target_gene": ["non-targeting"] * 40 + ["G6"] * 40,
            "perturbation_class": ["non-targeting"] * 40 + ["targeting"] * 40,
        },
        index=[f"cell{i}" for i in range(80)],
    )
    a = ad.AnnData(X=sp.csr_matrix(values), obs=obs, var=pd.DataFrame(index=["G6"]))
    a.layers[LOGNORM_LAYER] = sp.csr_matrix(np.log1p(values))
    a, table = _run(a)
    assert table.loc[0, "mean_ratio"] == pytest.approx(1.0)
    assert _statuses(a, "G6") == {"failed_group": 40}


def test_controls_and_ambiguous_cells_are_never_marked(adata):
    for mode in ("pooled", "per_context", "any_context"):
        a, _ = _run(adata.copy(), mode=mode, context_key="cell_line")
        ctrl = a.obs["perturbation_class"] == "non-targeting"
        amb = a.obs["perturbation_class"] == "ambiguous"
        assert (a.obs.loc[ctrl, OBS_KD_STATUS] == "control").all()
        assert (a.obs.loc[amb, OBS_KD_STATUS] == "untouched").all()
        assert a.obs.loc[ctrl | amb, OBS_KD_KEEP].all()
        assert a.obs.loc[ctrl | amb, OBS_KD_RATIO].isna().all()


def test_missing_context_column_fails_early(adata):
    with pytest.raises(ValueError, match="not an obs column"):
        _run(adata, mode="per_context", context_key="donor")


def test_context_modes_require_a_context_key():
    for mode in ("per_context", "any_context"):
        cfg = Config.from_dict({"input": {"h5ad": "x.h5ad"}, "knockdown_filter": {"enabled": True, "mode": mode}})
        with pytest.raises(ValueError, match="context_key"):
            cfg.validate()


# End-to-end on the synthetic lanes (condition: control / treated)


@pytest.mark.parametrize("method", ["mean_ratio", "count_model"])
def test_pipeline_writes_the_mask_without_removing_cells(tmp_path, method):
    import scanpy as sc
    from make_synthetic import KD_TARGETS, NULL_TARGETS, make_dataset
    from perturbseq_pipeline.cli import run_pipeline
    from test_pipeline import _base_config

    synthetic = {"dir": tmp_path, **make_dataset(tmp_path / "data", n_lanes=2, n_cells=300)}
    base = run_pipeline(_base_config(synthetic, tmp_path / "base"))
    kd = {"enabled": True, "mode": "per_context", "context_key": "condition", "min_cells": 5, "method": method}
    res = run_pipeline(_base_config(synthetic, tmp_path / "kd", knockdown_filter=kd))
    assert res.n_cells == base.n_cells
    obs = sc.read_h5ad(res.h5ad).obs
    for col in (OBS_KD_RATIO, OBS_KD_STATUS, OBS_KD_KEEP, "pert_log2fc_ntc", "pert_is_hit_ntc"):
        assert col in obs.columns
    table = pd.read_csv(res.tables["knockdown_filter"])
    assert set(table["context"]) == {"control", "treated"}
    by_target = table.groupby("target_gene")["group_status"].agg(set)
    for t in KD_TARGETS:
        assert by_target[t] == {"pass"}
    for t in NULL_TARGETS:
        assert by_target[t] == {"failed_group"}
    assert table["log2fc_ntc"].notna().all()


# method: count_model, on simulated negative-binomial counts with known truth


def _nb(rng, mean, phi):
    r = 1.0 / phi
    return rng.negative_binomial(r, r / (r + mean))


@pytest.fixture
def counts_adata():
    """600 controls and three targets of 300 cells, with varying library size.

    KD: 80% of cells at rho = 0.2, 20% escapers.   NULL: no knockdown.
    LOW: control mean 0.4 counts (about 30% detected), every cell at rho = 0.2.
    MOSTESC: 30% of cells at rho = 0.2, 70% escapers.
    """
    rng = np.random.default_rng(0)
    base = {"KD": 8.0, "NULL": 8.0, "LOW": 0.4, "MOSTESC": 8.0}
    n_ctrl, n_target, n_filler = 600, 300, 50
    labels = ["non-targeting"] * n_ctrl + [t for t in base for _ in range(n_target)]
    n = len(labels)
    depth = rng.lognormal(0.0, 0.3, n)
    level = np.ones((n, len(base)))
    escaper = np.zeros(n, dtype=bool)
    for j, t in enumerate(base):
        rows = np.flatnonzero(np.array(labels) == t)
        if t == "KD":
            escaper[rows[: int(0.2 * n_target)]] = True
            level[rows[int(0.2 * n_target) :], j] = 0.2
        if t == "LOW":
            level[rows, j] = 0.2
        if t == "MOSTESC":
            escaper[rows[: int(0.7 * n_target)]] = True
            level[rows[int(0.7 * n_target) :], j] = 0.2
    targets = np.column_stack([_nb(rng, base[t] * depth * level[:, j], 0.1) for j, t in enumerate(base)])
    filler = rng.poisson(40.0 * depth[:, None], (n, n_filler))
    counts = np.hstack([targets, filler]).astype(np.float64)
    library = counts.sum(axis=1, keepdims=True)
    lognorm = np.log1p(counts / library * np.median(library))
    obs = pd.DataFrame(
        {
            "target_gene": labels,
            "perturbation_class": ["non-targeting" if t == "non-targeting" else "targeting" for t in labels],
            "true_escaper": escaper,
        },
        index=[f"cell{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=list(base) + [f"F{k}" for k in range(n_filler)])
    a = ad.AnnData(X=sp.csr_matrix(lognorm), obs=obs, var=var)
    a.layers["counts"] = sp.csr_matrix(counts)
    a.layers[LOGNORM_LAYER] = sp.csr_matrix(lognorm)
    return a


def test_count_model_recovers_knockdown_and_escapers(counts_adata):
    a, table = _run(counts_adata, method="count_model")
    row = table.set_index("target_gene").loc["KD"]
    assert row["group_status"] == "pass"
    assert 0.12 < row["rho"] < 0.3
    assert 0.12 < row["escaper_fraction"] < 0.3
    # Escaper counts (mean 8) and knocked-down counts (mean 1.6) overlap, so many
    # escapers get a middling posterior and stay unmarked at 0.9. What the model
    # promises is that the posterior is honest: cells called are escapers, and
    # the posterior is high for escapers and low for knocked-down cells.
    kd = a.obs[a.obs["target_gene"] == "KD"]
    called = kd[OBS_KD_STATUS] == "escaper"
    assert called.sum() > 10
    assert kd.loc[called, "true_escaper"].mean() > 0.9
    prob = kd["kd_escaper_prob"]
    assert prob[kd["true_escaper"]].mean() > 0.5
    assert prob[~kd["true_escaper"]].mean() < 0.1


def test_count_model_fails_a_target_without_knockdown(counts_adata):
    """Without knockdown, rho and the escaper fraction are not identifiable (the
    fit may be rho near 1, or a low rho with most cells as escapers); either
    way the group must fail."""
    a, table = _run(counts_adata, method="count_model")
    assert table.set_index("target_gene").loc["NULL", "group_mean_ratio"] > 0.8
    assert _statuses(a, "NULL") == {"failed_group": 300}


def test_count_model_does_not_call_escapers_from_single_counts(counts_adata):
    """LOW is knocked down in every cell, but about 8% of cells still catch a
    count. The ratio cut calls those escapers; the count model, seeing that a
    single count is only weak evidence, calls almost none."""
    a_model, table = _run(counts_adata.copy(), method="count_model")
    a_ratio, _ = _run(counts_adata.copy(), method="mean_ratio")
    assert table.set_index("target_gene").loc["LOW", "group_status"] == "pass"
    n_model = _statuses(a_model, "LOW").get("escaper", 0)
    n_ratio = _statuses(a_ratio, "LOW").get("escaper", 0)
    assert n_model <= 0.03 * 300
    assert n_ratio > 3 * max(n_model, 1)


def test_count_model_rejects_normalized_counts(counts_adata):
    counts_adata.layers["counts"] = counts_adata.layers[LOGNORM_LAYER].copy()
    with pytest.raises(ValueError, match="raw counts"):
        _run(counts_adata, method="count_model")


def test_unknown_method_is_rejected():
    cfg = Config.from_dict({"input": {"h5ad": "x.h5ad"}, "knockdown_filter": {"enabled": True, "method": "median"}})
    with pytest.raises(ValueError, match="method"):
        cfg.validate()


def test_sharing_dispersion_across_genes_beats_gene_wise_with_few_controls():
    """400 genes, true phi = 0.05 + 0.5 / mean, only 30 control cells: each gene's
    own moment estimate is noisy, and shrinking toward the trend over genes
    brings it closer to the truth."""
    from perturbseq_pipeline.knockdown_filter import _control_fit

    rng = np.random.default_rng(1)
    n_cells, n_genes = 30, 400
    mean = np.exp(rng.uniform(np.log(0.5), np.log(50), n_genes))
    true_phi = 0.05 + 0.5 / mean
    s = rng.lognormal(0.0, 0.2, n_cells)
    s = s / np.median(s)
    counts = np.column_stack([_nb(rng, mean[g] * s, true_phi[g]) for g in range(n_genes)])
    mu, phi_shared = _control_fit(sp.csr_matrix(counts.astype(np.float64)), s)
    m = mu[None, :] * s[:, None]
    phi_gene = (((counts - m) ** 2).sum(0) - m.sum(0)) / (m**2).sum(0)

    def log_error(phi):
        return np.median(np.abs(np.log(np.maximum(phi, 1e-4)) - np.log(true_phi)))

    assert log_error(phi_shared) < 0.85 * log_error(phi_gene)


def test_dispersion_fit_stays_finite_with_ten_controls_and_tied_counts():
    """A lane with 10 controls (5 per split half): most genes have the same tiny
    total, so quantile bin edges coincide. Bins by quantile left a bin empty and
    turned every gene's phi into NaN, which then passed groups at rho = 0.01."""
    from perturbseq_pipeline.knockdown_filter import _control_fit

    rng = np.random.default_rng(2)
    n_cells = 10
    s = np.ones(n_cells)
    # 800 genes with one count in every cell (identical means in both halves), 200 others.
    tied = np.ones((n_cells, 800))
    expressed = np.column_stack([_nb(rng, np.full(n_cells, m), 0.1) for m in rng.uniform(0.5, 20, 200)])
    counts = sp.csr_matrix(np.hstack([tied, expressed]).astype(np.float64))
    mu, phi = _control_fit(counts, s)
    assert np.isfinite(phi[mu > 0]).all()


def test_count_model_fails_a_group_that_is_mostly_escapers(counts_adata):
    """MOSTESC is knocked down in only 30% of cells: rho is low, but most cells
    escape, so it fails. Without the escaper-fraction rule it would pass."""
    a, table = _run(counts_adata.copy(), method="count_model")
    row = table.set_index("target_gene").loc["MOSTESC"]
    assert row["rho"] < 0.3 and row["escaper_fraction"] > 0.5
    assert row["group_status"] == "failed_group"
    _, table = _run(counts_adata.copy(), method="count_model", max_escaper_fraction=1.0)
    assert table.set_index("target_gene").loc["MOSTESC", "group_status"] == "pass"
