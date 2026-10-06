"""Unit tests for the co-functional module / gene program stage.

These build a small AnnData with a *known* co-regulation structure — two groups
of perturbations, each driving a distinct block of genes in opposite directions
— and assert the stage recovers that structure (perturbations in a group share a
module; genes in a block share a program), plus that it skips gracefully when
underpowered.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from perturbseq_pipeline.config import Config
from perturbseq_pipeline.guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS, OBS_TARGET
from perturbseq_pipeline.modules import compute_modules


def _planted_adata(
    n_module_groups: int = 2,
    perts_per_group: int = 3,
    genes_per_block: int = 20,
    n_noise_genes: int = 20,
    cells_per_pert: int = 40,
    n_ntc: int = 60,
    add_noise_pert: int = 0,
    seed: int = 0,
):
    """AnnData where group g's perturbations raise gene-block g and lower the rest."""
    rng = np.random.default_rng(seed)
    n_blocks = n_module_groups
    block_genes = [[f"B{b}_G{j}" for j in range(genes_per_block)] for b in range(n_blocks)]
    noise_genes = [f"N{j}" for j in range(n_noise_genes)]
    genes = [g for blk in block_genes for g in blk] + noise_genes
    gene_idx = {g: i for i, g in enumerate(genes)}
    rows, targets, klass = [], [], []

    def _cell(active_block: int, sign: float):
        v = rng.normal(1.0, 0.15, size=len(genes))  # baseline (log-scale)
        for b, blk in enumerate(block_genes):
            level = 2.6 if (b == active_block and sign > 0) else (0.1 if b == active_block or sign < 0 else 1.0)
            # group g drives its own block up and the others down
            level = 2.6 if b == active_block else 0.15
            for g in blk:
                v[gene_idx[g]] = rng.normal(level, 0.15)
        return np.clip(v, 0, None)

    for grp in range(n_module_groups):
        for p in range(perts_per_group):
            name = f"P{grp}_{p}"
            for _ in range(cells_per_pert):
                rows.append(_cell(grp, +1))
                targets.append(name)
                klass.append(CLASS_TARGETING)
    # A low-cell-count perturbation with NO real effect (baseline, like NTC). A
    # hub metric gated only on |log2FC| would still flag many "DE genes" for it
    # from noise; the significance gate must not.
    for _ in range(add_noise_pert):
        v = rng.normal(1.0, 0.15, size=len(genes))
        rows.append(np.clip(v, 0, None))
        targets.append("PN")
        klass.append(CLASS_TARGETING)
    for _ in range(n_ntc):  # NTC: flat baseline
        v = rng.normal(1.0, 0.15, size=len(genes))
        rows.append(np.clip(v, 0, None))
        targets.append("non-targeting")
        klass.append(CLASS_NTC)
    X = np.asarray(rows, dtype=np.float64)
    adata = ad.AnnData(X=X.copy())
    adata.var_names = genes
    adata.layers["lognorm"] = X.copy()
    adata.obs[OBS_TARGET] = targets
    adata.obs[OBS_CLASS] = klass
    adata.obs["leiden"] = "0"
    # Only the planted block genes form the panel; noise genes are excluded.
    block_set = {g for blk in block_genes for g in blk}
    adata.var["highly_variable"] = [g in block_set for g in genes]
    return adata, block_genes


def _cfg(**modules):
    cfg = Config()
    cfg.modules.gene_selection = "hvg"  # use the planted genes directly
    cfg.modules.score_programs = False  # no leiden structure needed here
    cfg.modules.draw_networks = False
    cfg.modules.min_cells_per_perturbation = 10
    cfg.modules.min_perturbations = 5
    for k, v in modules.items():
        setattr(cfg.modules, k, v)
    return cfg


def test_modules_recovers_planted_coregulation():
    adata, block_genes = _planted_adata()
    cfg = _cfg(n_programs=2, n_modules=2)
    res = compute_modules(adata, cfg)
    assert res is not None
    assert res.n_modules == 2 and res.n_programs == 2
    assert res.effect_matrix.shape == (6, 40)  # 6 perturbations x 40 block genes
    # Perturbations of the same planted group must share a module.
    mod = res.modules.set_index("target_gene")["module"]
    g0 = {mod[f"P0_{p}"] for p in range(3)}
    g1 = {mod[f"P1_{p}"] for p in range(3)}
    assert len(g0) == 1 and len(g1) == 1, "each group should collapse to one module"
    assert g0 != g1, "the two groups must be different modules"
    # Genes of the same planted block must share a program.
    prog = res.gene_programs.set_index("gene")["program"]
    p0 = {prog[g] for g in block_genes[0]}
    p1 = {prog[g] for g in block_genes[1]}
    assert len(p0) == 1 and len(p1) == 1, "each block should collapse to one program"
    assert p0 != p1, "the two blocks must be different programs"


def test_modules_control_falls_back_to_other_without_ntc():
    adata, _ = _planted_adata(n_ntc=0)  # no non-targeting cells
    res = compute_modules(adata, _cfg(n_programs=2, n_modules=2))
    assert res is not None
    assert res.control == "other", "must fall back to the 'other' control with no NTC"


def test_modules_skips_when_underpowered():
    # Only three perturbations: below the default min_perturbations.
    adata, _ = _planted_adata(n_module_groups=1, perts_per_group=3)
    cfg = _cfg(n_programs=2, n_modules=2, min_perturbations=5)
    assert compute_modules(adata, cfg) is None


def test_hub_de_genes_are_significance_gated():
    """A low-cell-count perturbation with no real effect must not become a hub.

    Guards the cell-count confound: hub size is the number of genes that are BOTH
    large AND significant, so noise from a small group cannot inflate it.
    """
    adata, _ = _planted_adata(add_noise_pert=12)
    res = compute_modules(adata, _cfg(n_programs=2, n_modules=3, min_cells_per_perturbation=10))
    assert res is not None
    assert "n_cells" in res.hubs.columns, "hub table must expose n_cells"
    hubs = res.hubs.set_index("target_gene")
    assert hubs.loc["PN", "n_cells"] == 12
    # The no-effect, low-n perturbation must have essentially no DE genes...
    assert hubs.loc["PN", "n_de_genes"] <= 2, hubs.loc["PN", "n_de_genes"]
    # ...while the genuine perturbations show a large DE footprint.
    real = [f"P0_{p}" for p in range(3)] + [f"P1_{p}" for p in range(3)]
    assert max(int(hubs.loc[r, "n_de_genes"]) for r in real) >= 10


def test_module_program_strength_is_signed():
    adata, _ = _planted_adata()
    res = compute_modules(adata, _cfg(n_programs=2, n_modules=2))
    mp = res.module_program
    # Each module activates its own block's program (positive) and represses the
    # other (negative), so the strength matrix must contain both signs.
    vals = mp.to_numpy()
    assert np.nanmax(vals) > 0 and np.nanmin(vals) < 0
