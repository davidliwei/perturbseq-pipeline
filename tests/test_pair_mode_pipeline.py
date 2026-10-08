"""End-to-end pair-guide mode on synthetic two-lane MTX input (prefix cell ids, pair
reference, single-guide diagnostic, pair QC / ECDF / perturbation outputs)."""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from make_synthetic import make_split_lane  # noqa: E402

from perturbseq_pipeline import dual_guides as dg  # noqa: E402
from perturbseq_pipeline.cli import run_pipeline  # noqa: E402
from perturbseq_pipeline.config import Config  # noqa: E402
from perturbseq_pipeline.guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS, OBS_TARGET  # noqa: E402


def _read_mtx(d: Path):
    with gzip.open(d / "matrix.mtx.gz", "rb") as fh:
        m = scipy.io.mmread(fh).tocsr()  # features x cells
    with gzip.open(d / "barcodes.tsv.gz", "rt") as fh:
        bcs = [l.strip() for l in fh if l.strip()]
    with gzip.open(d / "features.tsv.gz", "rt") as fh:
        feats = pd.read_csv(fh, sep="\t", header=None)
    return m, bcs, feats


def _pair_up(guide_dir: Path, seed: int):
    """Give each cell a scaffold-C partner: same target (70 %), NTC (15 %) or another target (15 %)."""
    m, bcs, feats = _read_mtx(guide_dir)
    ids = feats[0].astype(str).tolist()
    target = {g: ("ntc" if g.startswith("non_targeting") else g.split("_P1P2_")[0]) for g in ids}
    k = {g: int(g.rsplit("_", 1)[1]) for g in ids}
    scaf = {g: ("A" if k[g] % 2 == 1 else "C") for g in ids}
    by_target_c = {}
    for g in ids:
        if scaf[g] == "C":
            by_target_c.setdefault(target[g], []).append(g)
    rng = np.random.default_rng(seed)
    X = m.T.tolil()  # cells x guides
    col = {g: i for i, g in enumerate(ids)}
    for c in range(X.shape[0]):
        row = X.rows[c]
        if not row:
            continue
        top = max(row, key=lambda j: X[c, j])
        g = ids[top]
        if scaf[g] != "A":
            continue
        u = rng.random()
        if u < 0.70:
            partner_t = target[g]
        elif u < 0.85:
            partner_t = "ntc"
        else:
            partner_t = rng.choice([t for t in by_target_c if t not in (target[g], "ntc")])
        partner = rng.choice(by_target_c[partner_t])
        X[c, col[partner]] = max(int(X[c, top] * rng.uniform(0.7, 1.3)), 3)
    X = X.tocsr().astype(int)
    with gzip.open(guide_dir / "matrix.mtx.gz", "wb") as fh:
        scipy.io.mmwrite(fh, sp.csr_matrix(X.T), field="integer")
    ref = pd.DataFrame(
        {
            "guide_id": ids,
            "designed_sequence": ["ACGT" * 5] * len(ids),
            "target_gene_name": [target[g] for g in ids],
            "scaffold": [scaf[g] for g in ids],
            "is_non_targeting": [target[g] == "ntc" for g in ids],
            "pair_id": "",
        }
    )
    return ref


def test_pair_mode_end_to_end(tmp_path):
    lanes, guide_dirs = {}, {}
    ref = None
    for i, lane in enumerate(["L1", "L2"]):
        paths = make_split_lane(tmp_path / "split", lane, n_cells=300, seed=i)
        lanes[lane] = paths["gex"]
        guide_dirs[lane] = paths["guides"]
        ref = _pair_up(Path(paths["guides"]), seed=i)
    ref_path = tmp_path / "pair_reference.csv"
    ref.to_csv(ref_path, index=False)
    meta = tmp_path / "meta.csv"
    pd.DataFrame({"lane_id": list(lanes), "sample": ["S1", "S2"], "condition": ["c1", "c1"]}).to_csv(meta, index=False)
    cfg = Config.from_dict(
        {
            "run": {"name": "pairmode", "outdir": str(tmp_path / "run_pair")},
            "input": {"mtx_dirs": lanes, "guide_mtx_dirs": guide_dirs, "cell_id_format": "prefix"},
            "metadata": {"file": str(meta)},
            "qc": {"min_genes_per_cell": 10, "min_genes_final": 50, "max_pct_mt": 100},
            "guides": {
                "assignment_mode": "pair",
                "pair_reference": str(ref_path),
                "scaffold_column": "auto",
                "pair_id_column": "auto",
                "sequence_column": "auto",
                "require_complete_pair": True,
                "unresolved_pair_policy": "exclude",
                "single_guide_diagnostic": True,
                "ntc_label": "ntc",
                "min_umi": 3,
                "dominance_ratio": 2.0,
            },
            "cluster": {"n_top_genes": 80, "n_pcs": 10},
            "perturbation": {"min_cells_per_target": 5, "top_n_report": 2},
            "enrichment": {"enabled": False},
            "ps_score": {"enabled": False},
            "lochness": {"enabled": False},
            "modules": {"enabled": False},
            "distance": {"enabled": False},
            "distance_space": {"enabled": False},
            "meta_analysis": {"enabled": False},
        }
    )
    cfg.validate()
    assert cfg.guides.assignment_mode == "dual_guide_pair"
    result = run_pipeline(cfg)
    obs = result.adata.obs
    # prefixed, unique, lane-preserving cell ids
    assert obs.index.is_unique and all(n.startswith(("L1_", "L2_")) for n in obs.index)
    assert set(obs["lane_id"].astype(str)) == {"L1", "L2"}
    # pair statuses drive the primary labels
    st = obs[dg.OBS_PAIR_STATUS].astype(str)
    assert (st == dg.STATUS_PAIR_TARGETING).sum() > 50 and (st == dg.STATUS_PAIR_NTC).sum() > 5
    assert ((st == dg.STATUS_PAIR_TARGETING) == (obs[OBS_CLASS].astype(str) == CLASS_TARGETING)).all()
    assert ((st == dg.STATUS_PAIR_NTC) == (obs[OBS_CLASS].astype(str) == CLASS_NTC)).all()
    assert (st == dg.STATUS_PAIR_TARGET_NTC).sum() > 0 and (
        obs.loc[st == dg.STATUS_PAIR_TARGET_NTC, OBS_CLASS].astype(str) == "ambiguous"
    ).all()
    assert (st == dg.STATUS_DUAL_TARGET).sum() > 0
    for c in (
        dg.OBS_SLOT_ID.format(c="A"),
        dg.OBS_SLOT_ID.format(c="C"),
        dg.OBS_PAIR_ID,
        dg.OBS_SG_CLASS,
        dg.OBS_SG_TARGET,
    ):
        assert c in obs.columns
    out = Path(cfg.run.outdir)
    for t in (
        "pair_assignment_per_lane",
        "pair_guide_qc_per_lane",
        "cell_counts_before_after",
        "single_guide_diagnostic_vs_pair",
        "pair_perturbation_by_target",
        "pair_perturbation_by_pair",
        "pair_perturbation_hit_counts_per_lane",
        "target_support_matrix",
        "cluster_composition",
        "cluster_sizes",
    ):
        assert (out / "tables" / f"{t}.csv").is_file(), t
    bt = pd.read_csv(out / "tables" / "pair_perturbation_by_target.csv")
    assert {
        "fdr_ks",
        "neg_log10_fdr",
        "log2fc",
        "n_target_pair_cells",
        "n_ntc_pair_cells",
        "assignment_stratum",
    } <= set(bt.columns)
    prim = bt[
        (bt.control == "ntc") & (bt.assignment_stratum == "primary_pair_targeting") & (bt.lane_id == "ALL")
    ].set_index("target_gene")
    assert prim.loc["KDGENE1", "is_hit"] and prim.loc["KDGENE1", "log2fc"] < 0
    assert not prim.loc["NULLGENE1", "is_hit"]
    cc = pd.read_csv(out / "tables" / "cell_counts_before_after.csv")
    assert cc.loc[cc.lane_id == "ALL", "final_cells_retained"].iloc[0] == result.n_cells
    for f in (
        "figures/guides/pair_assignment_status_counts_and_fractions.png",
        "figures/guides/single_guide_diagnostic_vs_pair_status.png",
        "figures/qc/cell_counts_before_after_per_lane.png",
        "figures/qc/ecdf_total_counts_before_after.png",
        "figures/clustering/umap_by_sample_condition_pair_status.png",
        "figures/clustering/pca_by_sample_condition_pair_status.png",
        "figures/clustering/cluster_composition_sample_pairstatus_class.png",
        "figures/perturbation/pair_volcano_target_level.png",
        "figures/perturbation/pair_heatmap_neg_log10_fdr_by_lane.png",
        "figures/perturbation/pair_ecdf_overview_top_targets.png",
        "figures/perturbation/ecdf/ecdf_KDGENE1.png",
        "report.html",
    ):
        assert (out / f).is_file() and (out / f).stat().st_size > 0, f
    html = (out / "report.html").read_text()
    assert "Pair-guide assignment is the primary label set" in html
    # scaffold classes round-trip into uns even with scaffold_column: auto
    assert "guide_scaffolds" in result.adata.uns and set(map(str, result.adata.uns["guide_scaffolds"])) <= {
        "A",
        "C",
        "unknown",
    }


def test_prefix_ids_single_lane_matches_combined_subset(tmp_path):
    from perturbseq_pipeline.io import load_data

    lanes, guide_dirs = {}, {}
    for i, lane in enumerate(["L1", "L2"]):
        paths = make_split_lane(tmp_path / "split", lane, n_cells=120, seed=i)
        lanes[lane], guide_dirs[lane] = paths["gex"], paths["guides"]
    meta = tmp_path / "meta.csv"
    pd.DataFrame({"lane_id": list(lanes), "sample": ["S1", "S2"]}).to_csv(meta, index=False)
    base = {
        "input": {"mtx_dirs": lanes, "guide_mtx_dirs": guide_dirs, "cell_id_format": "prefix"},
        "metadata": {"file": str(meta)},
    }
    both = load_data(Config.from_dict(base))
    one = load_data(
        Config.from_dict(
            {
                **base,
                "input": {**base["input"], "mtx_dirs": {"L1": lanes["L1"]}, "guide_mtx_dirs": {"L1": guide_dirs["L1"]}},
            }
        )
    )
    assert list(one.expr.obs_names) == [n for n in both.expr.obs_names if n.startswith("L1_")]
    assert both.guides.obs_names.equals(both.expr.obs_names)
    assert "cell_barcode" in both.expr.obs.columns


def test_guide_mtx_barcode_suffix_is_reconciled(tmp_path):
    """Bare guide barcodes vs '-1' expression barcodes must align, not error out."""
    from perturbseq_pipeline.io import load_data

    paths = make_split_lane(tmp_path / "split", "L1", n_cells=120, seed=0)
    gdir = Path(paths["guides"])
    with gzip.open(gdir / "barcodes.tsv.gz", "rt") as fh:
        bcs = [l.strip() for l in fh if l.strip()]
    with gzip.open(gdir / "barcodes.tsv.gz", "wt") as fh:
        fh.write("\n".join(b.rsplit("-", 1)[0] if "-" in b else b for b in bcs) + "\n")
    data = load_data(
        Config.from_dict({"input": {"mtx_dirs": {"L1": paths["gex"]}, "guide_mtx_dirs": {"L1": paths["guides"]}}})
    )
    assert data.guides is not None and data.guides.obs_names.equals(data.expr.obs_names)
    assert data.guides.X.sum() > 0
