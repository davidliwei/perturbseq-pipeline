"""Unit and integration tests for biological pathway enrichment of Stage 7 gene programs."""

from __future__ import annotations

from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import pytest

from perturbseq_pipeline.config import Config, ProgramEnrichmentConfig
from perturbseq_pipeline.gene_sets import (
    adapt_gene_set_to_species,
    benjamini_hochberg,
    clean_term_name,
    format_display_label,
    normalize_species_name,
    parse_gmt,
    run_ora_enrichment,
    run_program_enrichment,
    HALLMARK_GENE_SETS,
)
from perturbseq_pipeline.guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS, OBS_TARGET
from perturbseq_pipeline.modules import compute_modules


# Helpers & Synthetic Fixtures


def _planted_biological_adata():
    """Create an AnnData with known biological gene programs:

    P0: Interferon response genes (ISG15, IFIT1, IFIT2, MX1, OAS1, STAT1, ...)
    P1: Cell cycle / G2M genes (MKI67, TOP2A, CDK1, CCNB1, BUB1, AURKA, ...)
    """
    ifn_genes = ["ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1", "MX2", "OAS1", "OAS2", "STAT1", "STAT2", "IRF7", "RSAD2"]
    cc_genes = [
        "MKI67",
        "TOP2A",
        "CDK1",
        "CCNB1",
        "CCNA2",
        "BUB1",
        "AURKA",
        "PLK1",
        "CDC20",
        "CDC25C",
        "CHEK1",
        "BIRC5",
    ]
    noise_genes = [f"NOISE_{i}" for i in range(20)]
    all_genes = ifn_genes + cc_genes + noise_genes
    gene_idx = {g: i for i, g in enumerate(all_genes)}
    rng = np.random.default_rng(42)
    rows, targets, klass = [], [], []
    # IFN-activating perturbations (drive IFN genes UP, CC genes neutral/down)
    for name in ["IFNA", "IFNB", "IFNG"]:
        for _ in range(30):
            v = rng.normal(1.0, 0.1, size=len(all_genes))
            for g in ifn_genes:
                v[gene_idx[g]] = rng.normal(3.5, 0.2)
            rows.append(np.clip(v, 0, None))
            targets.append(name)
            klass.append(CLASS_TARGETING)
    # Proliferation-activating perturbations (drive CC genes UP, IFN genes neutral/down)
    for name in ["CCNA", "CCNB", "CCND"]:
        for _ in range(30):
            v = rng.normal(1.0, 0.1, size=len(all_genes))
            for g in cc_genes:
                v[gene_idx[g]] = rng.normal(3.5, 0.2)
            rows.append(np.clip(v, 0, None))
            targets.append(name)
            klass.append(CLASS_TARGETING)
    # NTC cells (flat baseline)
    for _ in range(50):
        v = rng.normal(1.0, 0.1, size=len(all_genes))
        rows.append(np.clip(v, 0, None))
        targets.append("non-targeting")
        klass.append(CLASS_NTC)
    X = np.asarray(rows, dtype=np.float64)
    adata = ad.AnnData(X=X.copy())
    adata.var_names = all_genes
    adata.layers["lognorm"] = X.copy()
    adata.obs[OBS_TARGET] = targets
    adata.obs[OBS_CLASS] = klass
    adata.obs["leiden"] = "0"
    adata.var["highly_variable"] = [g not in noise_genes for g in all_genes]
    return adata, ifn_genes, cc_genes, all_genes


# Unit Tests


def test_term_cleaner_and_display_labels():
    """Verify term name formatting removes technical prefixes and capitalizes cleanly."""
    assert clean_term_name("HALLMARK_INTERFERON_ALPHA_RESPONSE") == "Interferon Alpha Response"
    assert clean_term_name("REACTOME_CELL_CYCLE_CHECKPOINTS") == "Cell Cycle Checkpoints"
    assert clean_term_name("GOBP_DEFENSE_RESPONSE_TO_VIRUS") == "Defense Response To Virus"
    assert clean_term_name("KEGG_DNA_REPLICATION") == "DNA Replication"
    # Display label formatting
    assert format_display_label("P1", "Interferon Alpha Response") == "P1 — Interferon Alpha Response"
    assert format_display_label("P2", "unannotated") == "P2"
    assert format_display_label("P3", "no significant enrichment") == "P3"


def test_species_handling():
    """Verify safe handling of species strings and symbol casing."""
    assert normalize_species_name("human") == "human"
    assert normalize_species_name("Homo_sapiens") == "human"
    assert normalize_species_name("mouse") == "mouse"
    assert normalize_species_name("Mus_musculus") == "mouse"
    human_genes = ["ISG15", "IFIT1", "STAT1"]
    mouse_genes = adapt_gene_set_to_species(human_genes, "mouse")
    assert mouse_genes == ["Isg15", "Ifit1", "Stat1"]
    human_adapted = adapt_gene_set_to_species(["isg15", "ifit1"], "human")
    assert human_adapted == ["ISG15", "IFIT1"]


def test_ora_hypergeometric_exact_math():
    """Verify ORA hypergeometric calculation against theoretical known values."""
    # Universe of 100 genes, gene set has 20 genes.
    # Program has 10 genes, with 8 in the gene set.
    universe = [f"G_{i}" for i in range(100)]
    gene_set = [f"G_{i}" for i in range(20)]
    query = [f"G_{i}" for i in range(8)] + ["G_90", "G_91"]
    res = run_ora_enrichment(
        query_genes=query,
        universe_genes=universe,
        gene_sets={"TEST_SET": gene_set},
        source_name="test_source",
        source_version="v1.0",
        species="human",
        min_overlap=2,
        min_genes=5,
    )
    assert len(res) == 1
    hit = res[0]
    assert hit["overlap_count"] == 8
    assert hit["program_size"] == 10
    assert hit["gene_set_size"] == 20
    assert hit["background_size"] == 100
    assert hit["p_value"] < 1e-4
    assert hit["odds_ratio"] > 10.0


def test_benjamini_hochberg_monotonicity():
    """Verify multiple testing correction computes valid FDR values."""
    p_vals = np.array([0.001, 0.01, 0.02, 0.05, 0.5, 0.9])
    fdr = benjamini_hochberg(p_vals)
    assert len(fdr) == len(p_vals)
    assert np.all(fdr >= 0.0) and np.all(fdr <= 1.0)
    # Monotonicity check
    for i in range(len(fdr) - 1):
        assert fdr[i] <= fdr[i + 1]


def test_parse_custom_gmt(tmp_path):
    """Verify parsing user-provided GMT files."""
    gmt_file = tmp_path / "custom.gmt"
    content = (
        "CUSTOM_PATHWAY_1\thttps://example.org\tGENE1\tGENE2\tGENE3\nCUSTOM_PATHWAY_2\tDescription\tGENE4\tGENE5\n"
    )
    gmt_file.write_text(content)
    parsed = parse_gmt(gmt_file)
    assert len(parsed) == 2
    assert parsed["CUSTOM_PATHWAY_1"] == ["GENE1", "GENE2", "GENE3"]
    assert parsed["CUSTOM_PATHWAY_2"] == ["GENE4", "GENE5"]


# Stage 7 Integration Tests


def test_stage7_biological_enrichment_identifies_planted_pathways():
    """Verify Stage 7 correctly recovers IFN and Cell Cycle biological pathway annotations."""
    adata, ifn_genes, cc_genes, all_genes = _planted_biological_adata()
    cfg = Config()
    cfg.modules.gene_selection = "hvg"
    cfg.modules.score_programs = False
    cfg.modules.draw_networks = False
    cfg.modules.n_programs = 2
    cfg.modules.n_modules = 2
    cfg.modules.min_cells_per_perturbation = 10
    cfg.modules.min_perturbations = 4
    cfg.modules.program_enrichment.enabled = True
    cfg.modules.program_enrichment.sources = ["hallmark", "reactome", "go_bp"]
    cfg.modules.program_enrichment.fdr_alpha = 0.05
    res = compute_modules(adata, cfg)
    assert res is not None
    assert res.n_programs == 2
    assert res.n_modules == 2
    # Program enrichment table must be populated
    assert hasattr(res, "program_enrichment")
    assert not res.program_enrichment.empty
    enr = res.program_enrichment
    assert "program_id" in enr.columns
    assert "term" in enr.columns
    assert "fdr" in enr.columns
    assert "p_value" in enr.columns
    assert "overlap_genes" in enr.columns
    # Verify that the IFN program has significant Interferon response annotation
    # and the CC program has Cell Cycle / G2M annotation
    prog_df = res.gene_programs.set_index("gene")["program"]
    ifn_prog = prog_df[ifn_genes[0]]
    cc_prog = prog_df[cc_genes[0]]
    assert ifn_prog != cc_prog, "The two programs must be distinct"
    ifn_ann = res.program_annotations.get(ifn_prog, "")
    cc_ann = res.program_annotations.get(cc_prog, "")
    assert any(w in ifn_ann.lower() for w in ("interferon", "virus", "defense")), (
        f"Expected IFN-related label, got: {ifn_ann}"
    )
    assert any(w in cc_ann.lower() for w in ("cell cycle", "g2/m", "mitotic", "chromosome")), (
        f"Expected Cell Cycle-related label, got: {cc_ann}"
    )
    # Program IDs must remain P1, P2
    assert set(res.program_labels) == {"P1", "P2"}
    # Display labels must combine ID and biological annotation
    assert res.program_display_labels[ifn_prog].startswith(f"{ifn_prog} — ")
    assert res.program_display_labels[cc_prog].startswith(f"{cc_prog} — ")


def test_unannotated_fallback_when_no_enrichment():
    """Verify programs without significant biological enrichment remain 'unannotated'."""
    adata, _, _, _ = _planted_biological_adata()
    cfg = Config()
    cfg.modules.gene_selection = "hvg"
    cfg.modules.score_programs = False
    cfg.modules.draw_networks = False
    cfg.modules.n_programs = 2
    cfg.modules.n_modules = 2
    cfg.modules.min_cells_per_perturbation = 10
    cfg.modules.min_perturbations = 4
    cfg.modules.program_enrichment.enabled = True
    # Require an impossibly strict FDR alpha so nothing passes
    cfg.modules.program_enrichment.fdr_alpha = 1e-100
    res = compute_modules(adata, cfg)
    assert res is not None
    for p in res.program_labels:
        assert res.program_annotations[p] == "unannotated"
        assert res.program_display_labels[p] == p


def test_program_enrichment_can_be_disabled():
    """Verify program enrichment can be disabled without changing numerical Stage 7 outputs."""
    adata, _, _, _ = _planted_biological_adata()
    cfg_on = Config()
    cfg_on.modules.gene_selection = "hvg"
    cfg_on.modules.score_programs = False
    cfg_on.modules.draw_networks = False
    cfg_on.modules.n_programs = 2
    cfg_on.modules.n_modules = 2
    cfg_on.modules.min_cells_per_perturbation = 10
    cfg_on.modules.min_perturbations = 4
    cfg_on.modules.program_enrichment.enabled = True
    cfg_off = Config()
    cfg_off.modules.gene_selection = "hvg"
    cfg_off.modules.score_programs = False
    cfg_off.modules.draw_networks = False
    cfg_off.modules.n_programs = 2
    cfg_off.modules.n_modules = 2
    cfg_off.modules.min_cells_per_perturbation = 10
    cfg_off.modules.min_perturbations = 4
    cfg_off.modules.program_enrichment.enabled = False
    res_on = compute_modules(adata, cfg_on)
    res_off = compute_modules(adata, cfg_off)
    # Numerical outputs must be 100% IDENTICAL
    assert np.allclose(res_on.effect_matrix.to_numpy(), res_off.effect_matrix.to_numpy())
    assert np.allclose(res_on.module_program.to_numpy(), res_off.module_program.to_numpy())
    assert (res_on.gene_programs["program"] == res_off.gene_programs["program"]).all()
    assert (res_on.modules["module"] == res_off.modules["module"]).all()


def test_pipeline_end_to_end_program_enrichment(tmp_path):
    """Verify that a full pipeline run produces program_enrichment.csv, program_summary.csv, and report."""
    from perturbseq_pipeline.cli import run_pipeline

    adata, _, _, _ = _planted_biological_adata()
    h5ad_path = tmp_path / "bio_test.h5ad"
    adata.write_h5ad(h5ad_path)
    outdir = tmp_path / "results"
    cfg = Config.from_dict(
        {
            "run": {"name": "bio_test", "outdir": str(outdir)},
            "input": {"h5ad": str(h5ad_path), "counts_layer": "lognorm", "guide_obs_column": "target_gene"},
            "metadata": {"file": None},
            "guides": {"target_regex": r"^(.+)$"},
            "qc": {"min_genes_per_cell": 5, "min_genes_final": 5, "max_pct_mt": 100},
            "cluster": {"n_top_genes": 40, "n_pcs": 10},
            "perturbation": {"min_cells_per_target": 10, "top_n_report": 5},
            "modules": {
                "enabled": True,
                "gene_selection": "hvg",
                "n_programs": 2,
                "n_modules": 2,
                "min_cells_per_perturbation": 10,
                "min_perturbations": 4,
                "score_programs": False,
                "draw_networks": False,
                "program_enrichment": {
                    "enabled": True,
                    "species": "human",
                    "sources": ["hallmark", "reactome", "go_bp"],
                    "fdr_alpha": 0.05,
                    "top_terms_per_program": 5,
                },
            },
            "ps_score": {"enabled": False},
            "lochness": {"enabled": False},
            "distance": {"enabled": False},
            "distance_space": {"enabled": False},
        }
    )
    cfg.validate()
    result = run_pipeline(cfg)
    assert result.report.is_file(), "HTML report must be generated"
    html = result.report.read_text()
    assert "Co-functional modules &amp; gene programs" in html or "Co-functional modules & gene programs" in html
    assert (
        "Gene programs &amp; biological pathway annotations" in html
        or "Gene programs & biological pathway annotations" in html
    )
    # Tables must be written
    pe_csv = outdir / "tables" / "program_enrichment.csv"
    assert pe_csv.is_file(), "program_enrichment.csv must exist in results/tables"
    pe_df = pd.read_csv(pe_csv)
    assert not pe_df.empty
    assert "program_id" in pe_df.columns
    assert "term" in pe_df.columns
    assert "fdr" in pe_df.columns
    ps_csv = outdir / "tables" / "program_summary.csv"
    assert ps_csv.is_file(), "program_summary.csv must exist in results/tables"
    ps_df = pd.read_csv(ps_csv)
    assert not ps_df.empty
    assert "annotation" in ps_df.columns
    assert "display_label" in ps_df.columns
    # Figure must be produced
    dot_plot = outdir / "figures" / "modules" / "program_enrichment.png"
    assert dot_plot.is_file(), "program_enrichment.png dot plot must exist"
