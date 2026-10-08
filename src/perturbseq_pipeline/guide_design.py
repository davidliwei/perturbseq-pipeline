"""Designed-guide reference table for the basic QC stage.

Reads a design workbook / table (xlsx, xls, csv, tsv), auto-detects the
protospacer and target columns, recognises control guides through
configurable regexes, and produces one row per **designed** guide. Every
designed guide is retained whether or not it is later observed, so guide
count matrices always span the full reference.

The output frame has (at least) these columns::

    guide_id, protospacer, target, target_raw, is_control, scaffold,
    scaffold_source, design_index

plus every original workbook column (prefixed ``design_`` when it collides).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import Config, GuideDesignConfig

logger = logging.getLogger(__name__)

_PROTOSPACER_CANDIDATES = ("protospacer", "seq", "sequence", "spacer", "guide_sequence", "sgrna_sequence", "sgrna")
_TARGET_CANDIDATES = ("target", "gene", "target_gene", "target_gene_name", "gene_symbol", "symbol")
_ID_CANDIDATES = ("guide_id", "id", "sgrna_id", "guide", "name")
_SCAFFOLD_CANDIDATES = ("scaffold", "scaffold_class", "scaffold_id", "vector_position")

CONTROL_TARGET_LABEL = "control"


def _norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


def _pick_column(
    columns: Sequence[str], explicit: Optional[str], candidates: Sequence[str], what: str, required: bool
) -> Optional[str]:
    if explicit is not None:
        if explicit in columns:
            return explicit
        raise ValueError(f"guides.design.{what}_column={explicit!r} not found; columns: {list(columns)}")
    normed = {_norm(c): c for c in columns}
    for cand in candidates:
        if cand in normed:
            return normed[cand]
    if required:
        raise ValueError(
            f"Could not auto-detect the {what} column in the guide design table; "
            f"columns: {list(columns)}. Set guides.design.{what}_column."
        )
    return None


def read_design_table(path: str | Path, sheet: Optional[object] = None) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Guide design table not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    elif suffix in (".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t")
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported guide design table format: {path.suffix}")
    df.columns = [str(c) for c in df.columns]
    return df


def sanitize_label(label: str) -> str:
    """Make a target label safe for use inside an identifier."""
    s = re.sub(r"\s*\(([^)]*)\)", r"_\1", str(label).strip())
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_") or "NA"


def is_control_label(labels: Sequence[str], patterns: Sequence[str]) -> np.ndarray:
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    out = np.zeros(len(labels), dtype=bool)
    for i, lab in enumerate(labels):
        text = str(lab).strip()
        out[i] = any(c.search(text) for c in compiled)
    return out


def load_guide_design(cfg: Config) -> pd.DataFrame:
    """Build the designed-guide reference frame from ``guides.design``."""
    dcfg: GuideDesignConfig = cfg.guides.design
    if not dcfg.path:
        raise ValueError("guides.design.path is not set")
    raw = read_design_table(dcfg.path, dcfg.sheet)
    cols = list(raw.columns)
    proto_col = _pick_column(cols, dcfg.protospacer_column, _PROTOSPACER_CANDIDATES, "protospacer", True)
    target_col = _pick_column(cols, dcfg.target_column, _TARGET_CANDIDATES, "target", True)
    id_col = _pick_column(cols, dcfg.guide_id_column, _ID_CANDIDATES, "guide_id", False)
    scaffold_col = _pick_column(cols, dcfg.scaffold_column, _SCAFFOLD_CANDIDATES, "scaffold", False)
    df = pd.DataFrame(index=raw.index)
    proto = raw[proto_col].astype(str).str.strip()
    if dcfg.uppercase:
        proto = proto.str.upper()
    bad = ~proto.str.fullmatch(r"[ACGTN]+")
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} protospacer(s) in {dcfg.path} are not nucleotide strings, e.g. "
            f"{proto[bad].head(3).tolist()}"
        )
    df["protospacer"] = proto.to_numpy()
    df["target_raw"] = raw[target_col].astype(str).str.strip().to_numpy()
    patterns = dcfg.control_patterns if dcfg.control_patterns is not None else cfg.guides.ntc_patterns
    df["is_control"] = is_control_label(df["target_raw"].tolist(), patterns)
    df["target"] = np.where(df["is_control"], CONTROL_TARGET_LABEL, df["target_raw"])
    df["design_index"] = np.arange(len(df), dtype=int)
    if id_col is not None:
        ids = raw[id_col].astype(str).str.strip()
    else:
        counters: Dict[str, int] = {}
        ids_list: List[str] = []
        for lab in df["target_raw"]:
            key = sanitize_label(lab)
            counters[key] = counters.get(key, 0) + 1
            ids_list.append(dcfg.id_format.format(target=key, n=counters[key]))
        ids = pd.Series(ids_list, index=df.index)
    if ids.duplicated().any():
        dup = ids[ids.duplicated()].unique()[:5].tolist()
        raise ValueError(f"Guide ids are not unique in the design table (e.g. {dup}); set guides.design.id_format")
    df["guide_id"] = ids.to_numpy()
    dup_proto = df["protospacer"].duplicated(keep=False)
    if dup_proto.any():
        logger.warning(
            "%d protospacers are listed more than once in the design table; only the first "
            "occurrence can be matched: %s",
            int(dup_proto.sum()),
            df.loc[dup_proto, "protospacer"].unique()[:5].tolist(),
        )
    if scaffold_col is not None:
        df["scaffold"] = raw[scaffold_col].astype(str).str.strip().replace({"nan": "unknown", "": "unknown"}).to_numpy()
        df["scaffold_source"] = "design_table"
    else:
        df["scaffold"] = "unknown"
        df["scaffold_source"] = "unspecified"
    if dcfg.scaffold_table:
        st = read_design_table(dcfg.scaffold_table)
        st_proto = _pick_column(list(st.columns), None, _PROTOSPACER_CANDIDATES, "protospacer", True)
        st_scaf = _pick_column(list(st.columns), None, _SCAFFOLD_CANDIDATES, "scaffold", True)
        mapping = dict(zip(st[st_proto].astype(str).str.upper(), st[st_scaf].astype(str)))
        hit = df["protospacer"].map(mapping)
        df.loc[hit.notna(), "scaffold"] = hit[hit.notna()].to_numpy()
        df.loc[hit.notna(), "scaffold_source"] = "scaffold_table"
    # Keep the original columns for provenance.
    for c in cols:
        if c in (proto_col, target_col):
            continue
        name = _norm(c)
        if name in df.columns or not name:
            name = f"design_{name or 'col'}"
        vals = raw[c]
        if vals.dtype == object:
            vals = vals.astype(str)
        df[name] = vals.to_numpy()
    df = df.reset_index(drop=True)
    logger.info(
        "Guide design: %d designed guides, %d targets (+%d control guides) from %s "
        "[protospacer=%r target=%r id=%s scaffold=%s]",
        len(df),
        int(df.loc[~df["is_control"], "target"].nunique()),
        int(df["is_control"].sum()),
        dcfg.path,
        proto_col,
        target_col,
        id_col or "synthetic",
        scaffold_col or "n/a",
    )
    return df
