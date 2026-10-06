"""Normalization, dimensionality reduction, embedding and Leiden clustering.

After :func:`normalize`, the downstream contract is:

* ``layers['counts']``  — raw integer counts.
* ``layers['lognorm']`` — log1p library-size-normalized counts.
* ``adata.X``           — log-normalized values.

Large-data execution
--------------------
For multi-million-cell datasets, the standard Scanpy workflow can fail even
when the expression matrix is sparse. In particular,::

    sc.pp.scale(..., zero_center=True)

explicitly subtracts each gene mean. A sparse matrix then becomes dense.
For a dataset such as KOLF::

    2.66 million cells x 3,000 HVGs

one dense float64 matrix is already roughly 60 GiB, and scaling/PCA may require
multiple temporary matrices. This can exhaust hundreds of GiB of RAM.

The LARGE execution path therefore deliberately changes *how* standardized PCA
is computed without changing which cells enter the biological analysis:

1. HVGs are estimated from a reproducible bounded cell sample.
2. All cells are retained for PCA / neighbors / UMAP / Leiden.
3. The PCA working matrix contains HVGs only.
4. Sparse HVGs are variance-scaled with ``zero_center=False``.
5. PCA performs mean-centering internally/implicitly on the sparse matrix.
6. Explicit clipping is omitted in this sparse LARGE path because clipping
   uncentered values before PCA centering is not equivalent to clipping centered
   z-scores.
7. Full-expression covariate regression is never performed implicitly; when
   requested it is restricted to the HVG working object.

For STANDARD datasets the historical pipeline behaviour is preserved:
``sc.pp.scale`` performs its normal zero-centering and ``max_value`` clipping.

This module also avoids unnecessary full-matrix copies where possible and
stores Harmony output as float32 after correction.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from .compute import is_package_available, log_compute_decision, resolve_stage_backend
from .config import Config
from .io import LANE_KEY


logger = logging.getLogger(__name__)

LOGNORM_LAYER = "lognorm"
CLUSTER_KEY = "leiden"


# Small helpers


def _collect() -> None:
    """Request Python garbage collection after memory-heavy stages."""
    gc.collect()


def _matrix_gib(n_rows: int, n_cols: int, dtype=np.float64) -> float:
    """Dense memory footprint in GiB, used only for informative logging."""
    return int(n_rows) * int(n_cols) * np.dtype(dtype).itemsize / (1024.0**3)


# Normalization


def normalize(expr: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Library-size normalize and log1p-transform while preserving raw counts.

    STANDARD datasets retain the historical copy-safe behaviour.

    LARGE datasets avoid creating another complete copy merely to keep
    ``X`` and ``layers['lognorm']`` synchronized. Downstream large-data
    clustering never scales ``expr.X`` in place; scaling occurs only on the
    temporary HVG working object.
    """
    is_large = cfg.use_large_mode(expr.n_obs)
    # Already normalized
    if LOGNORM_LAYER in expr.layers:
        logger.info("Using pre-computed '%s' layer; skipping normalization", LOGNORM_LAYER)
        if is_large:
            expr.X = expr.layers[LOGNORM_LAYER]
            logger.info("Large-dataset memory mode: using '%s' directly as X without making a full copy", LOGNORM_LAYER)
        else:
            expr.X = expr.layers[LOGNORM_LAYER].copy()
        return expr
    # Preserve raw counts
    if "counts" not in expr.layers:
        logger.info("No 'counts' layer found; preserving current X as raw counts")
        expr.layers["counts"] = expr.X.copy()
        _collect()
    else:
        logger.info("Using existing 'counts' layer as raw expression")
    # Construct writable normalized expression
    #
    # Raw counts must remain unchanged, so normalization requires a matrix
    # distinct from layers['counts'].
    expr.X = expr.layers["counts"].copy()
    _collect()
    sc.pp.normalize_total(expr, target_sum=cfg.cluster.target_sum)
    sc.pp.log1p(expr)
    # Preserve normalized expression
    if is_large:
        # Intentionally avoid a second full expression-matrix copy.
        #
        # The downstream LARGE path never modifies expr.X in place.
        expr.layers[LOGNORM_LAYER] = expr.X
        logger.info(
            "Normalized to %s counts per cell and log1p-transformed "
            "[large-dataset memory mode; no duplicate lognorm copy]",
            cfg.cluster.target_sum or "median library size",
        )
    else:
        expr.layers[LOGNORM_LAYER] = expr.X.copy()
        logger.info(
            "Normalized to %s counts per cell and log1p-transformed", cfg.cluster.target_sum or "median library size"
        )
    _collect()
    return expr


# Highly variable genes


def _select_hvgs(expr: ad.AnnData, cfg: Config) -> None:
    """Select highly variable genes.

    STANDARD
        HVGs are estimated using the complete dataset, preserving the original
        pipeline behaviour.

    LARGE
        HVGs are estimated from a reproducible random sample of at most
        ``scaling.marker_max_cells`` cells. The resulting gene mask is then
        applied to the complete dataset.

    Importantly, the cell sample affects only feature selection. All cells are
    subsequently retained for PCA, neighbors, UMAP and Leiden.
    """
    c = cfg.cluster
    n_top = min(c.n_top_genes, expr.n_vars)
    is_large = cfg.use_large_mode(expr.n_obs)
    # Standard mode
    if not is_large:
        sc.pp.highly_variable_genes(expr, n_top_genes=n_top)
        n_selected = int(expr.var["highly_variable"].sum())
        if n_selected == 0:
            raise RuntimeError("HVG selection returned zero highly variable genes.")
        logger.info("Selected %d highly variable genes", n_selected)
        return
    # Large mode
    n_sample = min(int(cfg.scaling.marker_max_cells), expr.n_obs)
    logger.info("Large dataset detected: %d cells x %d genes", expr.n_obs, expr.n_vars)
    logger.info("Estimating %d HVGs using a reproducible subset of %d cells", n_top, n_sample)
    rng = np.random.default_rng(cfg.run.seed)
    sampled_idx = rng.choice(expr.n_obs, size=n_sample, replace=False)
    # Sorted sparse row slicing is generally cheaper and deterministic.
    sampled_idx.sort()
    hvg_expr = expr[sampled_idx, :].copy()
    sc.pp.highly_variable_genes(hvg_expr, n_top_genes=n_top)
    hvg_mask = hvg_expr.var["highly_variable"].to_numpy().astype(bool)
    n_selected = int(hvg_mask.sum())
    if n_selected == 0:
        raise RuntimeError("Large-dataset HVG selection returned zero highly variable genes.")
    expr.var["highly_variable"] = hvg_mask
    logger.info("Selected %d highly variable genes from %d-cell reference subset", n_selected, n_sample)
    del hvg_expr
    del sampled_idx
    del hvg_mask
    _collect()


# PCA


def _run_pca_on_hvgs(expr: ad.AnnData, cfg: Config) -> None:
    """Run PCA using only highly variable genes.

    STANDARD path
    -------------
    Preserves the previous implementation:

        HVG copy
          -> zero-centered scaling
          -> optional max-value clipping
          -> PCA

    LARGE sparse path
    -----------------
    Explicit zero-centering is forbidden because it densifies the matrix.

    Instead:

        sparse HVG copy
          -> divide each gene by its SD (zero_center=False)
          -> sparse-compatible PCA with implicit centering

    Algebraically, scaling by SD first and subtracting the scaled gene mean
    during PCA gives the same standardized coordinates as::

        (x - mean(x)) / sd(x)

    before PCA.

    ``scale_max_value`` clipping is deliberately NOT applied in this sparse
    path. Clipping ``x / sd`` before the mean is subtracted is not equivalent
    to clipping centered z-scores and would introduce a dataset-size-dependent
    statistical change.

    The full parent AnnData is never scaled.
    """
    c = cfg.cluster
    is_large = cfg.use_large_mode(expr.n_obs)
    if "highly_variable" not in expr.var.columns:
        raise RuntimeError("Cannot run PCA because highly variable genes have not been selected.")
    hvg_mask = expr.var["highly_variable"].to_numpy(dtype=bool)
    n_hvg = int(hvg_mask.sum())
    if n_hvg < 2:
        raise RuntimeError(f"Only {n_hvg} highly variable genes available; PCA requires at least 2.")
    n_pcs = int(min(c.n_pcs, expr.n_obs - 1, n_hvg - 1))
    if n_pcs < 1:
        raise RuntimeError(f"PCA resolved to n_pcs={n_pcs}; at least one component is required.")
    dense_gib32 = _matrix_gib(expr.n_obs, n_hvg, np.float32)
    dense_gib64 = _matrix_gib(expr.n_obs, n_hvg, np.float64)
    logger.info(
        "Creating PCA working matrix: %d cells x %d HVGs (dense equivalent %.1f GiB float32 / %.1f GiB float64)",
        expr.n_obs,
        n_hvg,
        dense_gib32,
        dense_gib64,
    )
    # This copy contains only the selected HVGs rather than all genes.
    pca_expr = expr[:, hvg_mask].copy()
    logger.info("PCA working matrix storage: %s", "sparse" if sparse.issparse(pca_expr.X) else "dense")
    # Optional covariate regression
    if c.regress_out:
        missing = [key for key in c.regress_out if key not in pca_expr.obs.columns]
        if missing:
            raise ValueError(f"cluster.regress_out refers to missing obs columns: {missing}")
        if is_large:
            # sc.pp.regress_out commonly creates dense intermediates.
            #
            # Even though we already restricted to HVGs, 2.6M x 3k can still
            # be tens of GiB when dense. Fail explicitly rather than allowing
            # a surprise OOM.
            raise ValueError(
                "cluster.regress_out is not supported in LARGE execution mode "
                "because Scanpy regression can densify the multi-million-cell "
                "HVG matrix. Remove cluster.regress_out for this run or perform "
                "the desired correction upstream. Batch correction through "
                "cluster.batch_key / Harmony remains supported."
            )
        logger.info("Regressing out %s on HVG working matrix", c.regress_out)
        sc.pp.regress_out(pca_expr, c.regress_out)
    # Scaling
    if c.scale_max_value is not None:
        if is_large and sparse.issparse(pca_expr.X):
            logger.info(
                "LARGE sparse scaling: zero_center=False to preserve sparsity; "
                "gene means will be centered by PCA. "
                "cluster.scale_max_value=%s is intentionally not applied in "
                "this path because clipping before centering is not equivalent "
                "to clipping centered z-scores.",
                c.scale_max_value,
            )
            # Critical KOLF fix:
            #
            # NEVER use zero_center=True here. It would turn the sparse
            # 2.66M x 3000 matrix into a dense matrix.
            #
            # We variance-scale while leaving zero entries sparse.
            sc.pp.scale(pca_expr, zero_center=False, max_value=None)
        else:
            logger.info("Scaling HVG working matrix with max_value=%s", c.scale_max_value)
            # Historical STANDARD behaviour.
            sc.pp.scale(pca_expr, zero_center=True, max_value=c.scale_max_value)
    elif is_large and sparse.issparse(pca_expr.X):
        logger.info("LARGE PCA: cluster.scale_max_value is null; running sparse PCA without prior variance scaling.")
    # PCA
    logger.info("Running PCA: n_comps=%d, solver=arpack, zero_center=True", n_pcs)
    # Scanpy supports mean-centered PCA for sparse matrices without requiring
    # callers to explicitly materialize a centered dense matrix. This is the
    # crucial difference from sc.pp.scale(zero_center=True).
    sc.tl.pca(pca_expr, n_comps=n_pcs, zero_center=True, svd_solver="arpack", random_state=cfg.run.seed)
    # Keep the cell embedding compact.
    expr.obsm["X_pca"] = np.asarray(pca_expr.obsm["X_pca"], dtype=np.float32)
    if "pca" in pca_expr.uns:
        expr.uns["pca"] = pca_expr.uns["pca"]
    logger.info("PCA: %d components on %d HVGs", n_pcs, n_hvg)
    del pca_expr
    del hvg_mask
    _collect()


# Embedding and clustering


def embed_and_cluster(expr: ad.AnnData, cfg: Config) -> ad.AnnData:
    """HVG selection, PCA, optional Harmony, neighbors, UMAP and Leiden."""
    c = cfg.cluster
    is_large = cfg.use_large_mode(expr.n_obs)
    sc.settings.seed = cfg.run.seed
    decision = resolve_stage_backend("clustering", cfg, n_cells=expr.n_obs)
    if cfg.compute.log_backend_decisions:
        log_compute_decision(decision)
    logger.info("Clustering execution mode: %s", "LARGE" if is_large else "STANDARD")
    # 1. Highly variable genes
    _select_hvgs(expr, cfg)
    # 2. PCA
    _run_pca_on_hvgs(expr, cfg)
    n_pcs = int(expr.obsm["X_pca"].shape[1])
    # 3. Optional Harmony
    use_rep = "X_pca"
    if c.batch_key:
        harmony_rep = _run_harmony(expr, cfg)
        if harmony_rep is not None:
            use_rep = harmony_rep
    # 4. Neighbor graph
    logger.info("Computing %d-nearest-neighbor graph using %s over %d cells", c.n_neighbors, use_rep, expr.n_obs)
    sc.pp.neighbors(expr, n_neighbors=c.n_neighbors, n_pcs=n_pcs, use_rep=use_rep, random_state=cfg.run.seed)
    _collect()
    # 5. UMAP
    #
    # This intentionally preserves full-data UMAP semantics.
    #
    # On multi-million-cell datasets this may itself become the next major
    # computational bottleneck. If that occurs, optimize it independently;
    # do not silently subsample here because downstream plots currently assume
    # one coordinate per analysed cell.
    logger.info("Computing UMAP for %d cells", expr.n_obs)
    sc.tl.umap(expr, min_dist=c.umap_min_dist, random_state=cfg.run.seed)
    _collect()
    # 6. Leiden
    logger.info("Running Leiden clustering at resolution %.2f", c.leiden_resolution)
    sc.tl.leiden(
        expr,
        key_added=CLUSTER_KEY,
        resolution=c.leiden_resolution,
        flavor="igraph",
        n_iterations=2,
        directed=False,
        random_state=cfg.run.seed,
    )
    n_clusters = int(expr.obs[CLUSTER_KEY].nunique())
    logger.info("Leiden clustering at resolution %.2f: %d clusters", c.leiden_resolution, n_clusters)
    # Restore X
    #
    # Parent X was never scaled. The temporary PCA object absorbed all scaling.
    if LOGNORM_LAYER in expr.layers:
        if is_large:
            expr.X = expr.layers[LOGNORM_LAYER]
            logger.info("Large-dataset memory mode: restored X from '%s' without making a full copy", LOGNORM_LAYER)
        else:
            expr.X = expr.layers[LOGNORM_LAYER].copy()
    _collect()
    return expr


# Reset embedding


def reset_embedding(expr: ad.AnnData) -> ad.AnnData:
    """Drop everything :func:`embed_and_cluster` produced, in place.

    Used when a subset of an already embedded object is re-embedded on its own
    under ``cluster.assigned_only``.

    ``counts`` and ``lognorm`` layers remain untouched.
    """
    for key in ("X_pca", "X_pca_harmony", "X_umap"):
        if key in expr.obsm:
            del expr.obsm[key]
    for key in list(expr.obsp.keys()):
        del expr.obsp[key]
    for key in ("pca", "neighbors", "umap", CLUSTER_KEY, f"{CLUSTER_KEY}_colors"):
        expr.uns.pop(key, None)
    if "highly_variable" in expr.var.columns:
        del expr.var["highly_variable"]
    if CLUSTER_KEY in expr.obs.columns:
        del expr.obs[CLUSTER_KEY]
    _collect()
    return expr


# Harmony


def _run_harmony(expr: ad.AnnData, cfg: Config) -> Optional[str]:
    """Batch-correct the PCA embedding with Harmony.

    Harmony operates only on the cells x PCs matrix, not on cells x genes.
    Its corrected result is stored as float32 after the algorithm completes to
    halve persistent embedding memory.
    """
    key = cfg.cluster.batch_key
    if key not in expr.obs.columns:
        raise ValueError(f"cluster.batch_key={key!r} is not an obs column. Available: {sorted(expr.obs.columns)[:30]}")
    n_batches = int(expr.obs[key].nunique())
    if n_batches < 2:
        logger.warning("cluster.batch_key=%r has a single level; skipping batch correction.", key)
        return None
    try:
        import harmonypy
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Harmony batch correction requested but harmonypy is not installed. "
            "Install it with: pip install 'perturbseq-pipeline[harmony]'"
        ) from exc
    # Harmony currently operates on a dense cells x PCs representation.
    #
    # This is acceptable relative to cells x genes:
    # 2.66M x 50 float64 is ~1 GiB rather than tens/hundreds of GiB.
    embedding = np.asarray(expr.obsm["X_pca"], dtype=np.float64)
    logger.info(
        "Running Harmony on embedding of shape %s across %d batches (dense float64 input approximately %.2f GiB)",
        embedding.shape,
        n_batches,
        embedding.nbytes / (1024.0**3),
    )
    out = harmonypy.run_harmony(embedding, expr.obs, key)
    corrected = np.asarray(out.result() if hasattr(out, "result") else out.Z_corr)
    # Orient defensively because harmonypy versions have historically differed
    # in whether corrected coordinates are cells x PCs or PCs x cells.
    if corrected.shape != embedding.shape:
        if corrected.T.shape == embedding.shape:
            corrected = corrected.T
        else:
            raise RuntimeError(
                "Harmony returned an embedding of shape "
                f"{corrected.shape}, which matches neither "
                f"{embedding.shape} nor its transpose."
            )
    expr.obsm["X_pca_harmony"] = corrected.astype(np.float32, copy=False)
    del corrected
    del embedding
    del out
    _collect()
    logger.info("Harmony batch correction on %r across %d batches", key, n_batches)
    return "X_pca_harmony"


# Summaries


def cluster_summary(expr: ad.AnnData, lane_key: str = LANE_KEY) -> pd.DataFrame:
    """Cluster sizes and lane composition.

    A cluster dominated by one lane can indicate residual batch structure, so
    per-lane composition is included whenever multiple lanes are available.
    """
    if CLUSTER_KEY not in expr.obs.columns:
        return pd.DataFrame()
    obs = expr.obs
    rows = []
    for cl, sub in obs.groupby(obs[CLUSTER_KEY].astype(str), observed=True):
        row = {"cluster": cl, "n_cells": len(sub), "pct_of_total": round(100.0 * len(sub) / max(expr.n_obs, 1), 2)}
        if "n_genes_by_counts" in sub.columns:
            row["median_genes"] = float(np.median(sub["n_genes_by_counts"]))
        if "pct_counts_mt" in sub.columns:
            row["median_pct_mt"] = round(float(np.median(sub["pct_counts_mt"])), 2)
        if lane_key in sub.columns and obs[lane_key].nunique() > 1:
            share = sub[lane_key].astype(str).value_counts(normalize=True)
            if not share.empty:
                row["top_lane"] = str(share.index[0])
                row["top_lane_pct"] = round(100.0 * float(share.iloc[0]), 1)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("n_cells", ascending=False).reset_index(drop=True)


def cluster_composition(expr: ad.AnnData, group_key: str) -> pd.DataFrame:
    """Contingency table of Leiden cluster against another ``obs`` column."""
    if CLUSTER_KEY not in expr.obs.columns or group_key not in expr.obs.columns:
        return pd.DataFrame()
    return expr.obs.groupby([CLUSTER_KEY, group_key], observed=True).size().unstack(fill_value=0)
