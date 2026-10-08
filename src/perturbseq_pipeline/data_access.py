"""Centralized storage and data access abstraction layer for perturbseq-pipeline.

This module decouples biological analysis modules from underlying storage formats
(in-memory AnnData, backed H5AD, or future Zarr) and provides zero-copy / shared-memory
data access utilities for multiprocessing workers.

Key design principles:
1. **Separation of Concerns**: Analysis modules request only the specific representation
   (embeddings, target gene expression, cell indices, obs columns) they need.
2. **Selective Materialization**: Never materializes full cell × gene dense matrices.
3. **Backed H5AD Support**: Handles disk-backed AnnData transparently for large screens.
4. **Shared Multiprocessing Representation**: Facilitates zero-copy memory mapping of
   low-dimensional embeddings (e.g. X_pca) across CPU workers.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .cluster import LOGNORM_LAYER
from .config import Config, StorageConfig
from .guides import CLASS_NTC, CLASS_TARGETING, OBS_CLASS, OBS_TARGET

logger = logging.getLogger(__name__)


# Storage Mode Resolution & Diagnostics


def is_backed(expr: ad.AnnData) -> bool:
    """Return True if the AnnData expression matrix is backed on disk."""
    return bool(getattr(expr, "isbacked", False))


def resolve_storage_mode(cfg: Config, n_cells: int = 0, is_h5ad_input: bool = True) -> str:
    """Determine effective storage access mode ('in_memory' or 'backed').

    Parameters
    ----------
    cfg : Config
        Pipeline configuration containing storage settings.
    n_cells : int
        Number of cells in the dataset.
    is_h5ad_input : bool
        True if the input is an H5AD file.

    Returns
    -------
    str
        'in_memory' or 'backed'.
    """
    mode = cfg.storage.mode.lower()
    if mode == "in_memory":
        return "in_memory"
    if mode == "backed":
        if is_h5ad_input:
            return "backed"
        logger.warning("[storage] Backed storage requested but input is not an H5AD file; using in-memory mode.")
        return "in_memory"
    # AUTO mode
    if is_h5ad_input and cfg.storage.prefer_backed_h5ad and n_cells >= cfg.storage.backed_threshold_cells:
        return "backed"
    return "in_memory"


def log_storage_diagnostics(
    input_format: str, access_mode: str, n_cells: int, n_vars: int, is_backed_obj: bool = False
) -> None:
    """Emit standardized concise runtime storage diagnostics."""
    logger.info(
        "[storage] input_format=%s access_mode=%s cells=%d genes=%d backed=%s",
        input_format,
        access_mode,
        n_cells,
        n_vars,
        "true" if is_backed_obj else "false",
    )


# Representation / Embedding Access


def get_embedding(
    expr: ad.AnnData, rep_name: str = "X_pca", cell_indices: Optional[np.ndarray] = None, dtype: type = np.float32
) -> np.ndarray:
    """Extract a low-dimensional embedding matrix (e.g. X_pca) safely.

    Parameters
    ----------
    expr : ad.AnnData
        AnnData object with embeddings in obsm.
    rep_name : str
        Key in expr.obsm to retrieve.
    cell_indices : Optional[np.ndarray]
        Optional 1D integer index array to slice.
    dtype : type
        Target NumPy dtype (default float32 for memory efficiency).

    Returns
    -------
    np.ndarray
        Extracted embedding array of shape (N, d).
    """
    if rep_name in expr.obsm:
        rep = expr.obsm[rep_name]
    else:
        # Fallbacks
        fallback = None
        for cand in ("X_pca_harmony", "X_pca"):
            if cand in expr.obsm:
                fallback = cand
                break
        if fallback is not None:
            logger.warning("[storage] Representation %r not found in obsm; using fallback %r", rep_name, fallback)
            rep = expr.obsm[fallback]
        else:
            raise ValueError(
                f"Requested representation {rep_name!r} not found in obsm (available: {sorted(expr.obsm.keys())})."
            )
    if sparse.issparse(rep):
        rep = rep.toarray()
    arr = np.asarray(rep, dtype=dtype)
    if cell_indices is not None:
        if len(cell_indices) == 0:
            return np.empty((0, arr.shape[1]), dtype=dtype)
        return arr[cell_indices]
    return arr


# Expression Layer & Gene Value Access


def get_layer(expr: ad.AnnData, layer_name: Optional[str] = LOGNORM_LAYER) -> Any:
    """Retrieve expression layer, falling back to expr.X if absent."""
    if layer_name and layer_name in expr.layers:
        return expr.layers[layer_name]
    return expr.X


def get_expression_vector(
    expr: ad.AnnData,
    gene: str,
    cell_indices: Optional[np.ndarray] = None,
    layer_name: Optional[str] = LOGNORM_LAYER,
    dtype: type = np.float32,
) -> np.ndarray:
    """Extract expression of a single gene for all cells or a specific cell subset.

    Never materializes full dense cell × gene matrices. Directly indexes sparse
    or backed matrices.

    Parameters
    ----------
    expr : ad.AnnData
        AnnData containing expression data.
    gene : str
        Target gene symbol / var_name.
    cell_indices : Optional[np.ndarray]
        Optional cell indices to extract. If None, extracts for all cells.
    layer_name : Optional[str]
        Layer to read from (default 'lognorm').
    dtype : type
        Return NumPy dtype (default float32).

    Returns
    -------
    np.ndarray
        1D array of expression values.
    """
    if gene not in expr.var_names:
        raise KeyError(f"Gene {gene!r} not found in var_names.")
    gene_idx = expr.var_names.get_loc(gene)
    layer = get_layer(expr, layer_name)
    if cell_indices is not None and len(cell_indices) == 0:
        return np.empty(0, dtype=dtype)
    if cell_indices is not None:
        # Sliced extraction
        if sparse.issparse(layer):
            # For CSR/CSC sparse matrices, slice specific rows for the target column
            col = layer[cell_indices, gene_idx]
            if sparse.issparse(col):
                col = col.toarray()
            return np.asarray(col, dtype=dtype).ravel()
        elif is_backed(expr):
            # Backed H5AD: slice cell_indices on the HDF5 dataset
            vals = layer[cell_indices, gene_idx]
            if sparse.issparse(vals):
                vals = vals.toarray()
            return np.asarray(vals, dtype=dtype).ravel()
        else:
            vals = layer[cell_indices, gene_idx]
            return np.asarray(vals, dtype=dtype).ravel()
    # All cells
    col = layer[:, gene_idx]
    if sparse.issparse(col):
        col = col.toarray()
    return np.asarray(col, dtype=dtype).ravel()


# Metadata & Target Mapping Access


def get_obs_column(expr: ad.AnnData, col_name: str) -> np.ndarray:
    """Retrieve a 1D NumPy array from AnnData.obs."""
    if col_name not in expr.obs.columns:
        raise KeyError(f"Column {col_name!r} not found in expr.obs.")
    return expr.obs[col_name].to_numpy()


def get_target_indices_map(expr: ad.AnnData, guide_class: str = CLASS_TARGETING) -> Dict[str, np.ndarray]:
    """Construct a map of {target_gene: np.ndarray[cell_indices]} once.

    Avoids allocating full boolean masks for every target gene repeatedly.
    """
    obs = expr.obs
    targets_col = obs[OBS_TARGET].astype(str).to_numpy()
    klass = obs[OBS_CLASS].astype(str).to_numpy()
    targeting_mask = (klass == guide_class) if guide_class else np.ones(len(klass), dtype=bool)
    targeting_indices = np.flatnonzero(targeting_mask)
    df = pd.DataFrame({"target": targets_col[targeting_indices], "cell_idx": targeting_indices})
    return {
        str(target): group["cell_idx"].to_numpy(dtype=np.int64, copy=True)
        for target, group in df.groupby("target", observed=True, sort=False)
    }


def get_control_indices(expr: ad.AnnData, cfg: Config, control_choice: str = "ntc") -> np.ndarray:
    """Retrieve 1D array of cell indices belonging to the specified control group."""
    klass = expr.obs[OBS_CLASS].astype(str).to_numpy()
    if control_choice == "ntc":
        return np.flatnonzero(klass == CLASS_NTC).astype(np.int64)
    elif control_choice == "other":
        return np.flatnonzero(klass == CLASS_TARGETING).astype(np.int64)
    else:
        # Check custom column if present
        if control_choice in expr.obs.columns:
            mask = expr.obs[control_choice].to_numpy(dtype=bool)
            return np.flatnonzero(mask).astype(np.int64)
        return np.flatnonzero(klass == CLASS_NTC).astype(np.int64)


# Shared Memory / Memmap Utilities for CPU Workers


class SharedArrayBuffer:
    """Memory-mapped or read-only shared array wrapper for zero-copy worker access."""

    def __init__(
        self, array: np.ndarray, create_memmap: bool = True, temp_dir: Optional[Union[str, Path]] = None
    ) -> None:
        self.shape = array.shape
        self.dtype = array.dtype
        self._temp_file: Optional[Path] = None
        self._memmap: Optional[np.memmap] = None
        if create_memmap:
            fd, path_str = tempfile.mkstemp(
                prefix="perturbseq_shm_", suffix=".dat", dir=str(temp_dir) if temp_dir else None
            )
            os.close(fd)
            self._temp_file = Path(path_str)
            mm = np.memmap(self._temp_file, dtype=self.dtype, mode="w+", shape=self.shape)
            mm[:] = array[:]
            mm.flush()
            del mm
            # Reopen in read-only mode
            self._memmap = np.memmap(self._temp_file, dtype=self.dtype, mode="r", shape=self.shape)
            self.array: np.ndarray = self._memmap
        else:
            self.array = np.asarray(array)

    def close(self) -> None:
        """Clean up the underlying memory-mapped file."""
        if self._memmap is not None:
            del self._memmap
            self._memmap = None
        if self._temp_file is not None and self._temp_file.exists():
            try:
                self._temp_file.unlink()
            except Exception:
                pass
            self._temp_file = None

    def __del__(self) -> None:
        self.close()
