"""Tests for the basic QC stage: loader, config, QC flags, Scrublet wrapper,
guide design parser, streaming guide counter, guide multiplet flags and the
two-well synthetic integration run.

The central invariant — flagged doublets / guide multiplets are annotated but
never removed — is asserted explicitly in several places.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from make_synthetic_basic_qc import SCAFFOLDS, TSO, basic_qc_config, make_dataset, make_design, write_10x_h5  # noqa: E402

from perturbseq_pipeline.config import Config  # noqa: E402
from perturbseq_pipeline import io as io_mod  # noqa: E402
from perturbseq_pipeline import qc as qc_mod  # noqa: E402
from perturbseq_pipeline.doublets import run_scrublet  # noqa: E402
from perturbseq_pipeline.guide_counting import (  # noqa: E402
    GuideCountJob,
    GuideReadSpec,
    build_protospacer_index,
    count_fastq_file,
    count_guides,
    read_guide_counts,
    write_guide_counts,
)
from perturbseq_pipeline.guide_design import load_guide_design  # noqa: E402
from perturbseq_pipeline.guide_qc import attach_guide_counts, infer_scaffold_classes  # noqa: E402
from perturbseq_pipeline.guide_counting import GuideCountResult  # noqa: E402


# Fixtures


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("basic_qc_synthetic")
    return make_dataset(out, wells=("W1", "W2"), n_cells=320)


@pytest.fixture(scope="session")
def qc_run(dataset, tmp_path_factory):
    from perturbseq_pipeline.cli import run_pipeline

    outdir = tmp_path_factory.mktemp("basic_qc_run")
    cfg = Config.from_dict(basic_qc_config(dataset, outdir))
    cfg.validate()
    result = run_pipeline(cfg, config_path="synthetic.yaml")
    return {"result": result, "cfg": cfg, "outdir": outdir}


def _base_cfg(**over) -> Config:
    data = {"run": {"stop_after": "qc"}, "samples": {"S1": {"gex_h5": "x.h5"}}}
    for k, v in over.items():
        data.setdefault(k, {}).update(v)
    return Config.from_dict(data)


# 10x loader


def test_read_10x_h5_preserves_counts_barcodes_and_ids(dataset):
    info = dataset["wells"]["W1"]
    adata = io_mod.read_10x_h5(info["h5"], "W1")
    assert adata.n_obs == info["n_cells"]
    assert sp.isspmatrix_csr(adata.X)
    assert np.issubdtype(adata.X.dtype, np.integer)
    assert np.issubdtype(adata.layers["counts"].dtype, np.integer)
    assert (adata.X != adata.layers["counts"]).nnz == 0
    assert list(adata.obs["cell_barcode"]) == info["barcodes"]
    assert adata.obs_names.is_unique
    assert adata.obs_names[0] == f"W1_{info['barcodes'][0]}"
    assert "gene_ids" in adata.var and adata.var["gene_ids"].iloc[0] == "ENSG00000000000"
    assert "feature_types" in adata.var and set(adata.var["feature_types"]) == {"Gene Expression"}
    assert adata.var_names.is_unique  # DUPGENE made unique
    assert "DUPGENE" in adata.var_names and "DUPGENE-1" in adata.var_names
    with pytest.raises(FileNotFoundError):
        io_mod.read_10x_h5(Path(info["h5"]).parent / "missing.h5", "W1")


def test_obs_names_are_globally_unique_across_wells(dataset):
    a = io_mod.read_10x_h5(dataset["wells"]["W1"]["h5"], "W1")
    b = io_mod.read_10x_h5(dataset["wells"]["W2"]["h5"], "W2")
    # force a shared barcode: the same raw barcode in two wells must not collide
    combined = ad.concat([a, b], join="inner", index_unique=None)
    assert combined.obs_names.is_unique
    assert combined.n_obs == a.n_obs + b.n_obs


# Configuration


def test_multi_sample_config_and_guide_pairing_parse(dataset, tmp_path):
    cfg = Config.from_dict(basic_qc_config(dataset, tmp_path))
    cfg.validate()
    samples = cfg.resolved_samples()
    assert list(samples) == ["W1", "W2"]
    assert samples["W1"].guide_library == "W1F"
    assert samples["W2"].guide_library == "W2F"
    assert samples["W1"].condition_code == "COND1" and samples["W2"].gem_well == "B"
    assert cfg.resolved_mode() == "samples"
    # YAML round trip keeps the samples block
    p = tmp_path / "cfg.yaml"
    cfg.dump_yaml(p)
    cfg2 = Config.from_yaml(p)
    assert cfg2.resolved_samples()["W2"].guide_fastq_dir == samples["W2"].guide_fastq_dir


def test_config_rejects_unknown_sample_keys_and_bad_stop_after():
    with pytest.raises(ValueError, match="Unknown config key"):
        _base_cfg(samples={"S1": {"gex_h5": "x.h5", "bogus": 1}})
    with pytest.raises(ValueError, match="stop_after"):
        _base_cfg(run={"stop_after": "cluster"}).validate()
    with pytest.raises(ValueError, match="stop_after: qc"):
        Config.from_dict({"samples": {"S1": {"gex_h5": "x.h5"}}}).validate()
    with pytest.raises(ValueError, match="guides.design.path"):
        _base_cfg(samples={"S1": {"gex_h5": "x.h5", "guide_fastq_dir": "d"}}).validate()
    cfg = _base_cfg()
    cfg.validate()  # GEX-only sample, no guides: valid
    assert cfg.guides.fastq.scaffolds == {"A": "GTTTAAGAGCTA", "C": "GTTTCAGAGCTA"}


def test_no_target_is_recognised_as_control_by_default():
    from perturbseq_pipeline.guide_design import is_control_label

    cfg = Config()
    flags = is_control_label(["NO-TARGET", "No_Target", "GENEA", "non-targeting", "NTC_1"], cfg.guides.ntc_patterns)
    assert flags.tolist() == [True, True, False, True, True]


# Expression QC


def _toy_expr(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(2.0, size=(n, 50)).astype(np.int32)
    X[:, -3:] = rng.poisson(0.5, size=(n, 3))
    X[0] = 0
    X[0, 0] = 1  # 1 count, 1 gene -> low
    X[1, -3:] = 500  # high mt
    adata = ad.AnnData(sp.csr_matrix(X))
    adata.var_names = [f"G{i}" for i in range(47)] + ["MT-A", "MT-B", "MT-C"]
    adata.obs_names = [f"c{i}" for i in range(n)]
    adata.obs["cell_barcode"] = adata.obs_names.to_numpy()
    adata.obs["condition_code"] = "X"
    return adata


def test_qc_metrics_thresholds_and_flags_are_deterministic_and_do_not_subset():
    cfg = _base_cfg(
        qc={
            "thresholds": {
                "method": "mad",
                "n_mads": 3,
                "min_genes_floor": 5,
                "min_counts_floor": 10,
                "max_pct_mt": 20.0,
            }
        }
    )
    adata = _toy_expr()
    n0 = adata.n_obs
    qc_mod.compute_basic_qc_metrics(adata, cfg)
    for col in ("total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo", "pct_counts_hb"):
        assert col in adata.obs
    assert adata.var["mt"].sum() == 3
    assert list(adata.var_names[-3:]) == ["MT-A", "MT-B", "MT-C"]  # names untouched
    t1 = qc_mod.resolve_sample_thresholds(adata.obs, cfg, "S", "X")
    t2 = qc_mod.resolve_sample_thresholds(adata.obs, cfg, "S", "X")
    assert t1 == t2
    assert t1["min_genes"] >= 5 and t1["min_counts"] >= 10
    assert t1["max_pct_mt"] == 20.0 and t1["method"] == "mad"
    qc_mod.flag_expression_qc(adata, t1)
    assert adata.n_obs == n0
    assert bool(adata.obs.loc["c0", "qc_low_counts"]) and bool(adata.obs.loc["c0", "qc_low_genes"])
    assert bool(adata.obs.loc["c1", "qc_high_mt"])
    assert not bool(adata.obs.loc["c0", "gex_qc_pass"]) and not bool(adata.obs.loc["c1", "gex_qc_pass"])
    passing = adata.obs[qc_mod.QC_FLAG_COLUMNS].sum(axis=1) == 0
    assert (adata.obs["gex_qc_pass"] == passing).all()


def test_fixed_method_condition_caps_and_overrides():
    cfg = _base_cfg(
        qc={
            "thresholds": {
                "method": "fixed",
                "min_genes_floor": 100,
                "min_counts_floor": 1000,
                "max_counts_ceiling": 50000,
                "max_pct_mt": 20.0,
                "max_pct_mt_by_condition": {"COND2": 15.0},
                "per_sample": {"S2": {"min_genes": 250}},
            }
        }
    )
    adata = _toy_expr()
    qc_mod.compute_basic_qc_metrics(adata, cfg)
    t = qc_mod.resolve_sample_thresholds(adata.obs, cfg, "S1", "COND1")
    assert (t["min_genes"], t["min_counts"], t["max_counts"], t["max_pct_mt"]) == (100, 1000, 50000, 20.0)
    assert t["max_genes"] is None
    t = qc_mod.resolve_sample_thresholds(adata.obs, cfg, "S2", "COND2")
    assert t["max_pct_mt"] == 15.0 and t["max_pct_mt_source"] == "condition:COND2"
    assert t["min_genes"] == 250 and t["min_genes_source"] == "per_sample_override"
    qc_mod.flag_expression_qc(adata, {"min_genes": None, "min_counts": None, "max_pct_mt": None})
    assert adata.obs["gex_qc_pass"].all()


# Scrublet wrapper


def test_scrublet_wrapper_scores_flags_and_keeps_every_cell(dataset):
    info = dataset["wells"]["W1"]
    cfg = _base_cfg(qc={"doublets": {"enabled": True, "threshold": 0.15}})
    adata = io_mod.read_10x_h5(info["h5"], "W1")
    X_before = adata.X.copy()
    n_before = adata.n_obs
    summary = run_scrublet(adata, cfg, "W1", seed=0)
    n_after = adata.n_obs
    assert n_before == n_after  # detection never removes cells
    assert "doublet_score" in adata.obs and "predicted_doublet" in adata.obs
    assert adata.obs["predicted_doublet"].dtype == bool
    assert np.isfinite(adata.obs["doublet_score"].to_numpy(dtype=float)).sum() > 0
    assert summary["n_predicted_doublets"] == int(adata.obs["predicted_doublet"].sum())
    assert (adata.X != X_before).nnz == 0  # raw counts untouched
    assert np.issubdtype(adata.X.dtype, np.integer)
    # the planted heterotypic doublets should score higher on average
    truth = info["is_doublet"]
    score = adata.obs["doublet_score"].to_numpy(dtype=float)
    assert np.nanmedian(score[truth]) > np.nanmedian(score[~truth])


def test_scrublet_disabled_writes_neutral_columns():
    cfg = _base_cfg(qc={"doublets": {"enabled": False}})
    adata = _toy_expr()
    run_scrublet(adata, cfg, "S")
    assert (~adata.obs["predicted_doublet"]).all() and adata.obs["doublet_score"].isna().all()


# Guide design


def test_guide_design_parser_keeps_all_designed_guides(dataset):
    cfg = _base_cfg(guides={"design": {"path": str(dataset["design_path"])}})
    design = load_guide_design(cfg)
    truth = dataset["design"]
    assert len(design) == len(truth) == 14
    assert design["guide_id"].is_unique
    assert design["is_control"].sum() == 5  # 4 sequenced + 1 unsequenced NO-TARGET
    assert (design.loc[design["is_control"], "target"] == "control").all()
    assert set(design.loc[~design["is_control"], "target"]) == {"GENEA", "GENEB", "GENEC", "GENED"}
    assert design["protospacer"].tolist() == truth["seq"].tolist()
    assert "on_target_efficacy_score" in design.columns and "beginswithg" in design.columns
    assert (design["scaffold"] == "unknown").all()  # no scaffold column in the workbook
    assert design["guide_id"].iloc[0] == "GENEA_1" and design["guide_id"].iloc[8] == "NO-TARGET_1"


def test_guide_design_explicit_columns_and_errors(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"sgRNA": ["ACGT" * 5, "TTTT" * 5], "Target": ["G1", "ntc"], "scaffold": ["A", "C"]}).to_csv(
        p, index=False
    )
    cfg = _base_cfg(guides={"design": {"path": str(p), "protospacer_column": "sgRNA", "target_column": "Target"}})
    d = load_guide_design(cfg)
    assert d["scaffold"].tolist() == ["A", "C"] and d["is_control"].tolist() == [False, True]
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"seq": ["ACGTN", "hello"], "gene": ["a", "b"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="not nucleotide"):
        load_guide_design(_base_cfg(guides={"design": {"path": str(bad)}}))


# Guide FASTQ parser


def _write_fastq(path: Path, seqs):
    with gzip.open(path, "wt") as fh:
        for i, s in enumerate(seqs):
            fh.write(f"@r{i}\n{s}\n+\n{'F' * len(s)}\n")


def test_guide_parser_handles_each_read_class(tmp_path):
    cfg = Config()
    spec = GuideReadSpec.from_config(cfg.guides.fastq)
    g1 = "AAACCCGGGTTTACGTACGA"  # guide 0 (scaffold A by construction)
    g2 = "TTGACCATGGACCTAGGCAT"  # guide 1 (scaffold C)
    exact, mm = build_protospacer_index([g1, g2], 0)
    bc_ok = "ACGTACGTACGTACGT"
    bc_bad = "GGGGGGGGGGGGGGGG"
    umi1, umi2, umi_n = "AAAAAAAAAAAA", "CCCCCCCCCCCC", "AAAAAAAAAAAN"
    A, C = SCAFFOLDS["A"], SCAFFOLDS["C"]
    reads = [
        f"{bc_ok}{umi1}{TSO}GG{g1}{A}",  # valid guide 0, scaffold A
        f"{bc_ok}{umi1}{TSO}GG{g1}{A}",  # duplicate UMI -> same molecule
        f"{bc_ok}{umi2}{TSO}GGG{g1}{A}",  # different UMI -> second molecule (3 Gs)
        f"{bc_ok}{umi1}{TSO}{g2}{C}",  # guide 1, scaffold C, zero Gs
        f"{bc_ok}{umi1}{TSO}GG{'ACGT' * 5}{A}",  # unknown guide
        f"{bc_bad}{umi1}{TSO}GG{g1}{A}",  # invalid barcode -> counted, not stored
        f"{bc_ok}{umi_n}{TSO}GG{g2}{C}",  # bad UMI
        "ACGT" * 37,  # no anchor
        f"{bc_ok}{umi2}{TSO}GG{g2}A{C}",  # 1-nt insertion before the anchor -> shifted exact match
    ]
    reads = [r[:151].ljust(60, "A") for r in reads]
    p = tmp_path / "t_R1_.fastq.gz"
    _write_fastq(p, reads)
    res = count_fastq_file(str(p), spec, exact, mm, {bc_ok.encode(): 0}, 2, decompressor="python")
    st = res["stats"]
    assert st["reads_total"] == 9
    assert st["reads_with_scaffold_anchor"] == 8
    assert st["reads_spacer_matched"] == 7 and st["reads_spacer_unmatched"] == 1
    assert st["reads_spacer_matched_via_shift"] == 1
    assert st["reads_matched_barcode_not_in_gex"] == 1
    assert st["reads_invalid_umi"] == 1
    assert st["reads_scaffold_A"] == 5 and st["reads_scaffold_C"] == 3
    assert st["reads_with_tso"] == 8
    # unique (cell, guide, UMI): g1/umi1, g1/umi2, g2/umi1, g2/umi2(shift) = 4
    assert res["codes"].size == 4
    assert res["guide_reads"].tolist() == [4, 3]
    assert res["guide_scaffold_reads"].tolist() == [[4, 0], [0, 3]]
    assert list(res["unmatched"].values()) == [1]
    # position_shift = 0 disables the positional fallback
    spec0 = GuideReadSpec.from_config(Config.from_dict({"guides": {"fastq": {"position_shift": 0}}}).guides.fastq)
    res0 = count_fastq_file(str(p), spec0, exact, mm, {bc_ok.encode(): 0}, 2, decompressor="python")
    assert res0["stats"]["reads_spacer_matched"] == 6 and res0["codes"].size == 3
    # explicit 1-mismatch mode
    exact1, mm1 = build_protospacer_index([g1, g2], 1)
    assert g1[:-1].encode() + b"T" in mm1 if g1[-1] != "T" else True
    _write_fastq(p, [f"{bc_ok}{umi1}{TSO}GG{g1[:-1]}{'T' if g1[-1] != 'T' else 'A'}{A}"[:151]])
    res1 = count_fastq_file(str(p), spec0, exact1, mm1, {bc_ok.encode(): 0}, 2, decompressor="python")
    assert res1["stats"]["reads_spacer_matched_via_mismatch"] == 1


def test_guide_counter_reproduces_synthetic_truth_and_writes_mtx(dataset, tmp_path):
    info = dataset["wells"]["W1"]
    cfg = Config.from_dict(basic_qc_config(dataset, tmp_path))
    design = load_guide_design(cfg)
    bare = [b.split("-")[0] for b in info["barcodes"]]
    job = GuideCountJob("W1", [str(f) for f in info["fastqs"]], bare)
    res = count_guides([job], design, cfg, n_workers=2)["W1"]
    assert res.counts.shape == (info["n_cells"], len(design))
    seq_to_col = {s: i for i, s in enumerate(design["protospacer"])}
    dense = res.counts.toarray()
    for i, bc in enumerate(bare):
        expected = np.zeros(len(design), dtype=int)
        for spacer, n in info["truth_umis"][bc].items():
            expected[seq_to_col[spacer]] = n
        assert dense[i].tolist() == expected.tolist(), f"cell {bc}"
    assert res.stats["reads_total"] == info["n_reads"]
    assert res.stats["reads_matched_barcode_not_in_gex"] == 300
    assert res.stats["n_files"] == 2
    # designed-but-unsequenced guides keep zero columns
    assert (dense[:, -2:] == 0).all()
    assert res.stats["guides_detected_any_umi"] == len(design) - 2
    paths = write_guide_counts(res, design, tmp_path / "gc")
    mat, bcs, gids = read_guide_counts(tmp_path / "gc", "W1")
    assert mat.shape == res.counts.shape and (mat != res.counts).nnz == 0
    assert bcs == bare and gids == design["guide_id"].tolist()
    summary = pd.read_csv(paths["guide_summary"], sep="\t")
    assert len(summary) == len(design)
    # empirical scaffold inference recovers the truth for every sequenced guide
    d2 = infer_scaffold_classes(design, {"W1": res}, cfg)
    truth = dataset["design"]["_scaffold_truth"]
    seq_ok = truth != "unsequenced"
    assert (d2.loc[seq_ok, "scaffold"].to_numpy() == truth[seq_ok].to_numpy()).all()
    assert (d2.loc[~seq_ok, "scaffold"] == "unknown").all()
    assert (d2.loc[seq_ok, "scaffold_source"] == "empirical").all()


# Guide multiplet logic


def _guide_fixture():
    design = pd.DataFrame(
        {
            "guide_id": ["A1", "A2", "C1", "C2", "U1"],
            "protospacer": ["AAAA", "AAAC", "CCCC", "CCCA", "GGGG"],
            "target": ["G1", "G2", "G1", "control", "G3"],
            "target_raw": ["G1", "G2", "G1", "NO-TARGET", "G3"],
            "is_control": [False, False, False, True, False],
            "scaffold": ["A", "A", "C", "C", "unknown"],
            "scaffold_source": ["design_table"] * 5,
            "design_index": range(5),
        }
    )
    counts = np.array(
        [
            [10, 0, 8, 0, 0],  # clean: 1A + 1C
            [10, 7, 8, 0, 0],  # 2 A -> multiplet
            [10, 0, 0, 0, 0],  # single A guide -> detected, structure fail, not multiplet
            [0, 0, 0, 0, 0],  # no guide
            [2, 0, 1, 0, 0],  # sub-threshold only
            [10, 0, 8, 9, 0],  # 2 C -> multiplet
            [10, 0, 8, 0, 5],  # unknown-scaffold guide -> structure fail
        ],
        dtype=np.int32,
    )
    adata = ad.AnnData(sp.csr_matrix(np.ones((7, 3), dtype=np.int32)))
    adata.obs_names = [f"S_bc{i}-1" for i in range(7)]
    adata.obs["cell_barcode"] = [f"bc{i}-1" for i in range(7)]
    res = GuideCountResult(
        sample_id="S",
        guide_ids=design["guide_id"].tolist(),
        cell_barcodes=[f"bc{i}" for i in range(7)],
        counts=sp.csr_matrix(counts),
        guide_reads=np.zeros(5, int),
        guide_scaffold_reads=np.zeros((5, 2), int),
        scaffold_names=["A", "C"],
    )
    return adata, res, design


def test_guide_multiplet_flags_but_never_subsets():
    adata, res, design = _guide_fixture()
    cfg = Config.from_dict({"guides": {"detection_threshold": 3}})
    n_before = adata.n_obs
    attach_guide_counts(adata, res, design, cfg)
    n_after = adata.n_obs
    assert n_before == n_after
    obs = adata.obs
    assert obs["n_guides"].tolist() == [2, 3, 1, 0, 0, 3, 3]
    assert obs["n_guides_A"].tolist() == [1, 2, 1, 0, 0, 1, 1]
    assert obs["n_guides_C"].tolist() == [1, 1, 0, 0, 0, 2, 1]
    assert obs["guide_umi_total"].tolist() == [18, 25, 10, 0, 3, 27, 23]
    assert obs["guide_detected"].tolist() == [True, True, True, False, False, True, True]
    assert obs["guide_multiplet_flag"].tolist() == [False, True, False, False, False, True, False]
    assert obs["guide_structure_pass"].tolist() == [True, False, False, False, False, False, False]
    assert (~obs["perturbation_assignable"]).all()
    assert obs["top_guide_A"].tolist()[:3] == ["A1", "A1", "A1"] and obs["top_guide_C"].iloc[5] == "C2"
    assert obs["top_guide"].iloc[3] == "none"
    assert adata.obsm["guide_counts"].shape == (7, 5)
    assert adata.uns["guide_names"] == design["guide_id"].tolist()


def test_guide_multiplet_without_scaffold_classes_uses_expected_guides_per_cell():
    adata, res, design = _guide_fixture()
    design = design.assign(scaffold="unknown")
    cfg = Config.from_dict({"guides": {"detection_threshold": 3, "multiplet": {"expected_guides_per_cell": 2}}})
    attach_guide_counts(adata, res, design, cfg)
    assert adata.obs["guide_multiplet_flag"].tolist() == [False, True, False, False, False, True, True]
    assert adata.obs["guide_structure_pass"].tolist() == [True, False, False, False, False, False, False]
    assert adata.n_obs == 7


# Synthetic two-well integration


def test_integration_outputs_exist(qc_run):
    r = qc_run["result"]
    b = r.basic_qc
    assert b is not None and r.execution_mode == "basic_qc"
    assert b.allcells_h5ad.is_file() and b.pass_h5ad.is_file() and b.report.is_file()
    for w in ("W1", "W2"):
        assert b.per_sample_h5ads[w].is_file()
        assert (b.guide_count_dirs[w] / "matrix.mtx.gz").is_file()
    for t in (
        "sample_qc_summary",
        "cell_qc_summary",
        "guide_qc_summary",
        "filtering_summary",
        "doublet_summary",
        "qc_thresholds",
        "scrublet_vs_guide_multiplet",
        "guide_feature_table",
    ):
        assert b.tables[t].is_file(), t
    figs = list((qc_run["outdir"] / "figures").rglob("*.png"))
    assert len(figs) >= 10
    assert (qc_run["outdir"] / "reports" / "provenance.json").is_file()
    assert (qc_run["outdir"] / "logs" / "resolved_config.yaml").is_file()
    assert "BASIC QC" in r.summary()


def test_integration_all_cells_retained_with_flags_aligned(qc_run, dataset):
    b = qc_run["result"].basic_qc
    allc = ad.read_h5ad(b.allcells_h5ad)
    n_input = sum(w["n_cells"] for w in dataset["wells"].values())
    assert allc.n_obs == n_input == b.n_cells_all
    assert allc.obs_names.is_unique and allc.var_names.is_unique
    assert np.issubdtype(allc.X.dtype, np.integer) and (allc.X != allc.layers["counts"]).nnz == 0
    assert set(allc.obs["sample_id"].astype(str)) == {"W1", "W2"}
    for col in (
        "sample_id",
        "condition_code",
        "gem_well",
        "guide_library",
        "cell_barcode",
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        "pct_counts_ribo",
        "qc_low_counts",
        "qc_high_counts",
        "qc_low_genes",
        "qc_high_genes",
        "qc_high_mt",
        "gex_qc_pass",
        "doublet_score",
        "predicted_doublet",
        "n_guides",
        "n_guides_A",
        "n_guides_C",
        "guide_umi_total",
        "guide_umi_A",
        "guide_umi_C",
        "guide_detected",
        "guide_structure_pass",
        "guide_multiplet_flag",
        "perturbation_assignable",
    ):
        assert col in allc.obs.columns, col
    for col in ("gene_ids", "feature_types", "mt", "ribo"):
        assert col in allc.var.columns
    # doublets exist and were retained
    n_dbl = int(allc.obs["predicted_doublet"].astype(bool).sum())
    assert n_dbl > 0 and n_dbl == b.n_predicted_doublets_all
    # guide multiplets exist and match construction
    gm = allc.obs["guide_multiplet_flag"].astype(bool)
    assert gm.sum() == sum(int(w["is_guide_multiplet"].sum()) for w in dataset["wells"].values())
    for w, info in dataset["wells"].items():
        sub = allc[allc.obs["sample_id"].astype(str) == w]
        assert list(sub.obs["cell_barcode"]) == info["barcodes"]  # order preserved
        assert sub.obs["guide_multiplet_flag"].to_numpy(dtype=bool).tolist() == info["is_guide_multiplet"].tolist()
        assert (~sub.obs["guide_detected"].to_numpy(dtype=bool) == info["no_guide"]).all()
        assert sub.obs["guide_library"].astype(str).iloc[0] == f"{w}F"
    # guide-count rows align with observations and the design
    G = allc.obsm["guide_counts"]
    assert G.shape == (allc.n_obs, 14)
    feats = allc.uns["guide_features"]
    assert len(feats) == 14 and list(feats["guide_id"]) == list(allc.uns["guide_names"])
    seq_to_col = {s: i for i, s in enumerate(feats["protospacer"])}
    dense = sp.csr_matrix(G).toarray()
    for w, info in dataset["wells"].items():
        idx = np.where(allc.obs["sample_id"].astype(str) == w)[0]
        for i, bc in zip(idx, info["barcodes"]):
            exp = np.zeros(14, int)
            for spacer, n in info["truth_umis"][bc.split("-")[0]].items():
                exp[seq_to_col[spacer]] = n
            assert dense[i].tolist() == exp.tolist()
    assert (allc.obs["guide_umi_total"].to_numpy() == dense.sum(axis=1)).all()
    # nothing normalised, no embeddings
    assert not [k for k in allc.obsm if k.startswith("X_")]
    assert "lognorm" not in allc.layers
    assert allc.uns["basic_qc"]["doublets_removed"] == 0
    assert "provenance" in allc.uns and "qc_thresholds" in allc.uns and "config" in allc.uns


def test_integration_expression_qc_object_keeps_doublets_and_multiplets(qc_run):
    b = qc_run["result"].basic_qc
    allc = ad.read_h5ad(b.allcells_h5ad)
    passed = ad.read_h5ad(b.pass_h5ad)
    assert passed.n_obs == int(allc.obs["gex_qc_pass"].sum()) == b.n_cells_pass
    assert passed.n_obs < allc.n_obs  # the planted low-quality cells drop out
    assert passed.obs["gex_qc_pass"].all()
    assert set(passed.obs_names) == set(allc.obs_names[allc.obs["gex_qc_pass"].to_numpy(dtype=bool)])
    expected_dbl = int((allc.obs["predicted_doublet"].astype(bool) & allc.obs["gex_qc_pass"]).sum())
    assert int(passed.obs["predicted_doublet"].astype(bool).sum()) == expected_dbl == b.n_predicted_doublets_pass > 0
    expected_gm = int((allc.obs["guide_multiplet_flag"].astype(bool) & allc.obs["gex_qc_pass"]).sum())
    assert int(passed.obs["guide_multiplet_flag"].astype(bool).sum()) == expected_gm == b.n_guide_multiplets_pass > 0
    # cells without guides survive expression QC when their transcriptome is fine
    assert (~passed.obs["guide_detected"].astype(bool)).sum() > 0
    assert passed.obsm["guide_counts"].shape == (passed.n_obs, 14)
    assert np.issubdtype(passed.X.dtype, np.integer)


def test_integration_tables_are_consistent(qc_run):
    b = qc_run["result"].basic_qc
    ss = pd.read_csv(b.tables["sample_qc_summary"], sep="\t")
    fs = pd.read_csv(b.tables["filtering_summary"], sep="\t")
    ds = pd.read_csv(b.tables["doublet_summary"], sep="\t")
    thr = pd.read_csv(b.tables["qc_thresholds"], sep="\t")
    assert ss["sample_id"].tolist() == ["W1", "W2"]
    assert (ss["cells_removed_in_this_stage"] == 0).all()
    assert ss["input_cells"].sum() == b.n_cells_all and ss["gex_qc_pass"].sum() == b.n_cells_pass
    assert ss["predicted_doublet"].sum() == b.n_predicted_doublets_all
    assert fs.loc[fs["sample_id"] == "ALL", "predicted_doublets_in_qc_object"].iloc[0] == b.n_predicted_doublets_pass
    assert (fs["doublets_removed"] == 0).all() and (fs["guide_multiplets_removed"] == 0).all()
    assert len(ds) == 2 and ds["threshold"].notna().all()
    assert set(thr["sample_id"]) == {"W1", "W2"} and thr["min_genes"].notna().all()
    cell = pd.read_csv(b.tables["cell_qc_summary"], sep="\t")
    assert len(cell) == b.n_cells_all


def test_per_sample_checkpoint_matches_input(qc_run, dataset):
    b = qc_run["result"].basic_qc
    info = dataset["wells"]["W2"]
    a = ad.read_h5ad(b.per_sample_h5ads["W2"])
    assert a.n_obs == info["n_cells"]
    assert list(a.obs["cell_barcode"]) == info["barcodes"]
    assert a.obsm["guide_counts"].shape == (info["n_cells"], 14)
    assert "scrublet" in a.uns and "qc_thresholds" in a.uns
    assert a.obs["predicted_doublet"].astype(bool).sum() > 0
    assert a.obs["guide_multiplet_flag"].astype(bool).sum() == int(info["is_guide_multiplet"].sum())


def test_gex_only_sample_runs_without_guides(dataset, tmp_path):
    """GEX-only wells (no guide data yet) go through the same stage."""
    from perturbseq_pipeline.cli import run_pipeline

    info = dataset["wells"]["W1"]
    cfg = Config.from_dict(
        {
            "run": {"name": "gex_only", "outdir": str(tmp_path / "out"), "stop_after": "qc"},
            "samples": {"W1": {"gex_h5": str(info["h5"]), "condition_code": "C"}},
            "qc": {"thresholds": {"min_genes_floor": 50, "min_counts_floor": 200}, "doublets": {"threshold": 0.15}},
            "output": {"h5ad_name": "gex_only_qc.h5ad"},
            "report": {"figure_dpi": 50},
        }
    )
    cfg.validate()
    r = run_pipeline(cfg)
    a = ad.read_h5ad(r.unfiltered_h5ad)
    assert a.n_obs == info["n_cells"]
    assert (~a.obs["guide_detected"]).all() and (~a.obs["guide_multiplet_flag"]).all()
    assert "guide_counts" not in a.obsm
    assert a.obs["predicted_doublet"].sum() > 0
    assert r.h5ad.is_file() and r.report.is_file()


def test_precomputed_guide_matrix_source(dataset, tmp_path, qc_run):
    """A guide count matrix written by the counter can be consumed as input."""
    from perturbseq_pipeline.cli import run_pipeline

    b = qc_run["result"].basic_qc
    info = dataset["wells"]["W1"]
    cfg = Config.from_dict(
        {
            "run": {"name": "matrix_in", "outdir": str(tmp_path / "out"), "stop_after": "qc"},
            "samples": {
                "W1": {"gex_h5": str(info["h5"]), "guide_matrix": str(b.guide_count_dirs["W1"]), "guide_library": "W1F"}
            },
            # the counter's guide summary doubles as a protospacer -> scaffold table
            "guides": {
                "design": {
                    "path": str(dataset["design_path"]),
                    "scaffold_table": str(b.guide_count_dirs["W1"] / "W1_guide_summary.tsv"),
                },
                "detection_threshold": 3,
            },
            "qc": {"thresholds": {"min_genes_floor": 50, "min_counts_floor": 200}, "doublets": {"enabled": False}},
            "output": {"h5ad_name": "m.h5ad"},
            "report": {"figure_dpi": 50},
        }
    )
    cfg.validate()
    r = run_pipeline(cfg)
    a = ad.read_h5ad(r.unfiltered_h5ad)
    ref = ad.read_h5ad(b.per_sample_h5ads["W1"])
    assert a.obsm["guide_counts"].shape == ref.obsm["guide_counts"].shape
    assert (sp.csr_matrix(a.obsm["guide_counts"]) != sp.csr_matrix(ref.obsm["guide_counts"])).nnz == 0
    assert a.obs["guide_multiplet_flag"].tolist() == ref.obs["guide_multiplet_flag"].tolist()
    assert a.obs["guide_structure_pass"].tolist() == ref.obs["guide_structure_pass"].tolist()
    assert a.uns["guide_source"] == "matrix"
    assert set(a.uns["guide_features"]["scaffold_source"]) <= {"scaffold_table", "unspecified"}


# Depth-aware detection rule, sensitivity table, threshold plausibility


def test_relative_detection_rule_and_sensitivity_table():
    from perturbseq_pipeline.guide_qc import detected_mask, guide_detection_sensitivity

    adata, res, design = _guide_fixture()
    counts = res.counts
    absolute = detected_mask(counts, 3).toarray()
    assert absolute.sum(axis=1).tolist() == [2, 3, 1, 0, 0, 3, 3]
    relative = detected_mask(counts, 3, 0.75).toarray()  # a guide needs >= 75% of the cell's top guide
    assert relative.sum(axis=1).tolist() == [2, 2, 1, 0, 0, 3, 2]  # cell 6: 8 >= 0.75*10, 5 < 7.5
    cfg = Config.from_dict({"guides": {"detection_threshold": 3, "multiplet": {"detection_min_fraction_of_top": 0.75}}})
    attach_guide_counts(adata, res, design, cfg)
    assert adata.obs["n_guides"].tolist() == [2, 2, 1, 0, 0, 3, 2]
    assert adata.obs["guide_structure_pass"].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        True,
    ]  # cell 1: A2 (7) drops below 7.5
    assert adata.obs["guide_multiplet_flag"].tolist() == [False, False, False, False, False, True, False]
    assert adata.n_obs == 7
    table = guide_detection_sensitivity(counts, design, cfg, "S")
    assert len(table) == 5 * 5 and set(table["sample_id"]) == {"S"}
    assert table["is_current_rule"].sum() == 0  # 0.75 is not on the default grid
    row = table[(table["detection_threshold_umi"] == 3) & (table["min_fraction_of_top"] == 0.0)].iloc[0]
    assert row["frac_guide_multiplet_flag"] == pytest.approx(2 / 7)
    with pytest.raises(ValueError, match="detection_min_fraction_of_top"):
        _base_cfg(guides={"multiplet": {"detection_min_fraction_of_top": 1.5}}).validate()


def test_scrublet_threshold_plausibility_assessment():
    from perturbseq_pipeline.doublets import assess_threshold

    rng = np.random.default_rng(0)
    scores = rng.beta(2, 20, size=10_000)
    ok = assess_threshold(scores, 0.25, int((scores > 0.25).sum()), "auto")
    assert not ok["threshold_suspect"] and "score_p99_9" in ok
    bad = assess_threshold(scores, 0.9, 0, "auto")
    assert bad["threshold_suspect"] and "99.9th" in bad["threshold_suspect_reason"]
    few = assess_threshold(scores, float(np.percentile(scores, 99.8)), 20, "auto")
    assert few["threshold_suspect"] and "0.5%" in few["threshold_suspect_reason"]
    manual = assess_threshold(scores, 0.9, 0, "manual")
    assert not manual["threshold_suspect"]


def test_guide_counter_scaffold_specific_features(dataset, tmp_path):
    """Features = designed spacer x scaffold class; UMIs land in the (spacer, true scaffold) column only."""
    from perturbseq_pipeline.guide_counting import FEATURE_SEP, expand_design_by_scaffold

    info = dataset["wells"]["W1"]
    cfg = Config.from_dict(basic_qc_config(dataset, tmp_path))
    cfg.guides.fastq.scaffold_specific_features = True
    design = load_guide_design(cfg)
    bare = [b.split("-")[0] for b in info["barcodes"]]
    job = GuideCountJob("W1", [str(f) for f in info["fastqs"]], bare)
    res = count_guides([job], design, cfg, n_workers=2)["W1"]
    names = list(cfg.guides.fastq.scaffolds)
    assert res.counts.shape == (info["n_cells"], 2 * len(design))
    assert res.feature_design is not None and len(res.feature_design) == 2 * len(design)
    fd = res.feature_design
    assert list(fd["guide_id"]) == [f"{g}{FEATURE_SEP}{c}" for g in design["guide_id"] for c in names]
    assert list(fd["guide_id"]) == res.guide_ids
    col = {(s, c): i for i, (s, c) in enumerate(zip(fd["protospacer"], fd["scaffold"]))}
    truth_scaf = dataset["design"].set_index("seq")["_scaffold_truth"]
    dense = res.counts.toarray()
    for i, bc in enumerate(bare):
        expected = np.zeros(dense.shape[1], dtype=int)
        for spacer, n in info["truth_umis"][bc].items():
            expected[col[(spacer, truth_scaf[spacer])]] = n
        assert dense[i].tolist() == expected.tolist(), f"cell {bc}"
    # collapsing the two scaffold columns reproduces the scaffold-agnostic count matrix
    res0 = count_guides([job], design, Config.from_dict(basic_qc_config(dataset, tmp_path)), n_workers=2)["W1"]
    collapsed = dense.reshape(dense.shape[0], len(design), len(names)).sum(axis=2)
    assert (collapsed == res0.counts.toarray()).all()
    assert res.stats["features"] == 2 * len(design) and res.stats["scaffold_specific_features"] is True
    paths = write_guide_counts(res, design, tmp_path / "gc_split")
    mat, bcs, gids = read_guide_counts(tmp_path / "gc_split", "W1")
    assert mat.shape == res.counts.shape and gids == res.guide_ids
    summary = pd.read_csv(paths["guide_summary"], sep="\t")
    assert len(summary) == 2 * len(design) and {"design_guide_id", "scaffold", "designed_slot"} <= set(summary.columns)
    # expansion helper keeps the design's own scaffold declaration when it has one
    d = design.copy()
    d["scaffold"] = ["A"] + ["unknown"] * (len(d) - 1)
    ex = expand_design_by_scaffold(d, names)
    assert ex["designed_slot"].tolist()[:2] == [True, False]
