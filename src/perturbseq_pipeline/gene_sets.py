"""Biological pathway enrichment and functional annotation for gene programs.

Over-Representation Analysis (hypergeometric test with BH correction, run by
gseapy) against MSigDB collections (Hallmark, Reactome, GO Biological Process,
KEGG; human or mouse, downloaded through gseapy) and user-supplied GMT files,
with the background restricted to the Stage 7 genes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import gseapy

logger = logging.getLogger(__name__)


# Term Name Beautifier

_ACRONYMS = {
    "Dna": "DNA",
    "Rna": "RNA",
    "Mrna": "mRNA",
    "Ifn": "IFN",
    "G2m": "G2/M",
    "E2f": "E2F",
    "Myc": "MYC",
    "Kras": "KRAS",
    "Mtor": "mTOR",
    "Mtorc1": "mTORC1",
    "Pi3k": "PI3K",
    "Akt": "AKT",
    "Tgfb": "TGF-beta",
    "Tgf": "TGF",
    "Nfkb": "NF-kB",
    "Tnf": "TNF",
    "Jak": "JAK",
    "Stat": "STAT",
    "Stat3": "STAT3",
    "Stat5": "STAT5",
    "Il2": "IL-2",
    "Il6": "IL-6",
    "P53": "p53",
    "Atp": "ATP",
    "Tca": "TCA",
    "Ros": "ROS",
    "Er": "ER",
    "Upr": "UPR",
    "Uv": "UV",
    "G1": "G1",
    "G2": "G2",
    "S": "S",
    "M": "M",
    "Rtk": "RTK",
    "Rtks": "RTKs",
    "Mapk": "MAPK",
    "Wnt": "Wnt",
    "Nod": "NOD",
    "Rig": "RIG-I",
    "Toll": "Toll",
    "Tlr": "TLR",
}


def clean_term_name(term: str) -> str:
    """Turn a database term into a clean, human-readable label.

    Examples:
        HALLMARK_INTERFERON_ALPHA_RESPONSE -> Interferon Alpha Response
        REACTOME_CELL_CYCLE_CHECKPOINTS -> Cell Cycle Checkpoints
        GOBP_DEFENSE_RESPONSE_TO_VIRUS -> Defense Response To Virus
        KEGG_DNA_REPLICATION -> DNA Replication
    """
    cleaned = term
    for prefix in ("HALLMARK_", "REACTOME_", "GOBP_", "GO_", "KEGG_", "BIOCARTA_", "PID_", "WP_"):
        if cleaned.upper().startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    # Replace underscores and hyphens with spaces
    words = re.split(r"[_\s]+", cleaned.strip())
    formatted_words = []
    for w in words:
        if not w:
            continue
        title_w = w.capitalize()
        # Fix known acronyms
        formatted = _ACRONYMS.get(title_w, title_w)
        # Handle embedded acronyms or special patterns
        if w.upper() in {"DNA", "RNA", "IFN", "MYC", "E2F", "KRAS", "TP53", "P53", "ATP", "TCA", "ROS", "MAPK", "WNT"}:
            formatted = _ACRONYMS.get(title_w, w.upper())
        formatted_words.append(formatted)
    return " ".join(formatted_words)


def format_display_label(program_id: str, annotation: str) -> str:
    """Format a compact display label for heatmaps and plots."""
    if not annotation or annotation.lower() in ("unannotated", "none", "no significant enrichment"):
        return program_id
    # Truncate if very long
    ann = annotation
    if len(ann) > 28:
        ann = ann[:25] + "..."
    return f"{program_id} — {ann}"


# MSigDB collection behind each gene-set source, per species. MSigDB has no mouse KEGG
# collection, so a mouse "kegg" source needs a GMT file in custom_gmt_files.
MSIGDB_CATEGORIES: Dict[str, Dict[str, str]] = {
    "human": {"hallmark": "h.all", "reactome": "c2.cp.reactome", "go_bp": "c5.go.bp", "kegg": "c2.cp.kegg_legacy"},
    "mouse": {"hallmark": "mh.all", "reactome": "m2.cp.reactome", "go_bp": "m5.go.bp"},
}
MSIGDB_SPECIES_SUFFIX = {"human": "Hs", "mouse": "Mm"}


# Program Enrichment Orchestrator

ENRICHMENT_COLUMNS = [
    "program_id",
    "annotation",
    "gene_set_source",
    "term",
    "clean_term",
    "program_size",
    "gene_set_size",
    "overlap_count",
    "overlap_genes",
    "p_value",
    "fdr",
    "odds_ratio",
    "background_size",
    "species",
    "gene_set_version",
]


def run_program_enrichment(
    program_genes: Dict[str, List[str]], universe: List[str], cfg: Any, species: str
) -> Tuple[pd.DataFrame, Dict[str, str], pd.DataFrame, Dict[str, str]]:
    """Over-representation of each Stage 7 gene program in the configured gene sets (gseapy).

    Parameters
    ----------
    program_genes : Dict[str, List[str]]
        Mapping of program ID (e.g. 'P1', 'P2') to member gene symbols.
    universe : List[str]
        Background gene universe (the genes eligible/included in Stage 7 effect matrix).
    cfg : ProgramEnrichmentConfig
        Configuration object containing sources, msigdb_version, fdr_alpha, top_terms_per_program, etc.
    species : str
        'human' or 'mouse' (``input.species``); picks the MSigDB collections.

    Returns
    -------
    enrichment_df : pd.DataFrame
        Full enrichment results table.
    program_annotations : Dict[str, str]
        Mapping from program ID -> biological annotation name (or 'unannotated').
    program_summary : pd.DataFrame
        Compact summary table with program ID, biological annotation, top term, FDR, size, top genes.
    display_labels : Dict[str, str]
        Mapping from program ID -> display label (e.g. 'P1 — Interferon response').
    """
    universe = [str(g) for g in universe if g]
    universe_set = set(universe)
    # Gene sets cut down to the background; only sets with min_genes..max_genes background genes are tested.
    gene_sets: Dict[str, List[str]] = {}
    term_source: Dict[str, str] = {}
    term_version: Dict[str, str] = {}
    for src in cfg.sources:
        key = src.lower().strip()
        if key in cfg.custom_gmt_files:
            gsets = gseapy.read_gmt(cfg.custom_gmt_files[key])
            version = f"Custom GMT ({Path(cfg.custom_gmt_files[key]).name})"
        elif key in MSIGDB_CATEGORIES[species]:
            category = MSIGDB_CATEGORIES[species][key]
            dbver = f"{cfg.msigdb_version}.{MSIGDB_SPECIES_SUFFIX[species]}"
            gsets = gseapy.Msigdb.get_gmt(category=category, dbver=dbver)
            version = f"MSigDB {dbver} {category}"
        else:
            logger.warning("Unrecognized gene-set source %r; skipping.", src)
            continue
        for term, genes in gsets.items():
            in_universe = sorted(universe_set.intersection(genes))
            if cfg.min_genes <= len(in_universe) <= cfg.max_genes:
                gene_sets[term] = in_universe
                term_source[term] = src
                term_version[term] = version
    tables: List[pd.DataFrame] = []
    program_annotations: Dict[str, str] = {}
    display_labels: Dict[str, str] = {}
    summary_rows: List[Dict[str, Any]] = []
    for prog_id, p_genes in program_genes.items():
        query = [g for g in p_genes if g in universe_set]
        # gseapy returns an empty list, not a table, when no gene set shares a gene with the program.
        res = []
        if query:
            res = gseapy.enrich(
                gene_list=query, gene_sets=gene_sets, background=universe, outdir=None, no_plot=True
            ).results
        table = pd.DataFrame(columns=ENRICHMENT_COLUMNS)
        if len(res):
            # gseapy's BH covers every term sharing >= 1 gene with the program, all sources together;
            # terms below min_overlap are dropped afterwards.
            overlap = res["Overlap"].str.split("/", expand=True).astype(int)
            table = pd.DataFrame(
                {
                    "program_id": prog_id,
                    "gene_set_source": res["Term"].map(term_source),
                    "term": res["Term"],
                    "clean_term": res["Term"].map(clean_term_name),
                    "program_size": len(query),
                    "gene_set_size": overlap[1],
                    "overlap_count": overlap[0],
                    "overlap_genes": res["Genes"].map(lambda g: ", ".join(sorted(g.split(";")))),
                    "p_value": res["P-value"],
                    "fdr": res["Adjusted P-value"],
                    "odds_ratio": res["Odds Ratio"],
                    "background_size": len(universe_set),
                    "species": species,
                    "gene_set_version": res["Term"].map(term_version),
                }
            )
            table = table[table["overlap_count"] >= cfg.min_overlap].sort_values("p_value", ignore_index=True)
            tables.append(table)
        significant = table[table["fdr"] <= cfg.fdr_alpha]
        best = significant.iloc[0] if len(significant) else None
        annotation = best["clean_term"] if best is not None else "unannotated"
        program_annotations[prog_id] = annotation
        display_labels[prog_id] = format_display_label(prog_id, annotation)
        summary_rows.append(
            {
                "program_id": prog_id,
                "annotation": annotation,
                "display_label": display_labels[prog_id],
                "top_term": best["term"] if best is not None else "None",
                "gene_set_source": best["gene_set_source"] if best is not None else "None",
                "fdr": best["fdr"] if best is not None else np.nan,
                "program_size": len(query),
                "top_genes": ", ".join(p_genes[:8]),
                "member_genes": ", ".join(p_genes),
            }
        )
    enrichment_df = pd.DataFrame(columns=ENRICHMENT_COLUMNS)
    if tables:
        enrichment_df = pd.concat(tables, ignore_index=True)
        enrichment_df["annotation"] = enrichment_df["program_id"].map(program_annotations)
        enrichment_df = enrichment_df[ENRICHMENT_COLUMNS].sort_values(["program_id", "p_value"], ignore_index=True)
    return enrichment_df, program_annotations, pd.DataFrame(summary_rows), display_labels
