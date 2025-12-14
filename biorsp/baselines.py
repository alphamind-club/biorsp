"""Baseline analyses for spatial gene expression."""

import logging

logger = logging.getLogger(__name__)
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse
from scipy.stats import f
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

from .geometry import cartesian_to_polar

EPSILON = 1e-9


def compute_morans_i(
    adata: AnnData,
    genes: Optional[Iterable[str]] = None,
    embedding_key: str = "X_umap",
    k: int = 30,
) -> pd.Series:
    """Compute Moran's I for spatial autocorrelation on the embedding graph.

    Returns a Series indexed by :class:`AnnData.var_names`."""
    if genes is None:
        genes = adata.var_names

    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=k)

    try:
        morans = sc.metrics.morans_i(adata)
        if isinstance(morans, np.ndarray):
            morans = pd.Series(morans, index=adata.var_names)
        return morans[genes] if genes is not None else morans
    except AttributeError:
        return _morans_i_custom(adata, genes)


def _morans_i_custom(adata: AnnData, genes: Iterable[str]) -> pd.Series:
    """Fallback Moran's I implementation used when scanpy's function is unavailable."""
    w = adata.obsp["connectivities"]

    results = {}
    x = adata.X
    if issparse(x):
        x = x.toarray()

    n = adata.n_obs
    w_sum = w.sum()

    for gene in genes:
        idx = adata.var_names.get_loc(gene)
        x = x[:, idx]
        x_bar = np.mean(x)
        z = x - x_bar

        num = n * (z.T @ w @ z)
        den = w_sum * np.sum(z**2)

        morans_index = num / (den + EPSILON)
        results[gene] = morans_index

    return pd.Series(results)


def compute_gradient_magnitude(
    adata: AnnData,
    genes: Iterable[str],
    embedding_key: str = "X_umap",
) -> pd.Series:
    """Estimate gradient magnitude of expression on the embedding.

    Uses local linear fits over neighbors to estimate per-cell gradient
    magnitudes and returns the mean per gene."""
    coords = adata.obsm[embedding_key]
    x = adata.X
    if issparse(x):
        x = x.toarray()

    results = {}

    nbrs: NearestNeighbors = NearestNeighbors(n_neighbors=30).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    for gene in genes:
        idx = adata.var_names.get_loc(gene)
        expr = x[:, idx]

        grads = []
        for i in range(adata.n_obs):
            nbr_indices = indices[i]
            local_coords = coords[nbr_indices] - coords[i]
            local_expr = expr[nbr_indices] - expr[i]

            grad, _, _, _ = np.linalg.lstsq(local_coords, local_expr, rcond=None)
            grads.append(np.linalg.norm(grad))

        results[gene] = np.mean(grads)

    return pd.Series(results)


def compute_neighborhood_enrichment(
    adata: AnnData,
    genes: Iterable[str],
    embedding_key: str = "X_umap",
    k: int = 30,
) -> pd.Series:
    """Compute enrichment of expression in local neighborhoods vs global background.
    Returns the maximum enrichment score observed across all neighborhoods."""
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=k)

    connectivities = adata.obsp["connectivities"]
    adj = (connectivities > 0).astype(float)

    x = adata.X
    if issparse(x):
        x = x.toarray()

    results = {}

    for gene in genes:
        idx = adata.var_names.get_loc(gene)
        expr = x[:, idx]

        degrees = np.array(adj.sum(axis=1)).flatten()
        local_means = (adj @ expr) / (degrees + EPSILON)

        global_mean = np.mean(expr)

        enrichment = np.max(local_means) / (global_mean + EPSILON)
        results[gene] = enrichment

    return pd.Series(results)


def run_baselines(
    adata: AnnData,
    genes: Iterable[str],
    embedding_key: str = "X_umap",
) -> pd.DataFrame:
    """Run baseline analyses and return a combined DataFrame with metrics."""
    logger.info("Computing Moran's I...")
    morans = compute_morans_i(adata, genes, embedding_key)

    logger.info("Computing Gradient Magnitude...")
    grads: pd.Series = compute_gradient_magnitude(adata, genes, embedding_key)

    logger.info("Computing Neighborhood Enrichment...")
    enrich: pd.Series = compute_neighborhood_enrichment(adata, genes, embedding_key)

    df = pd.DataFrame({"Morans_I": morans, "Gradient_Mag": grads, "Nb_Enrich": enrich})

    return df


def compute_morans_i_residualized(
    adata: AnnData,
    genes: Iterable[str],
    covariates: np.ndarray,
    embedding_key: str = "X_umap",
    k: int = 30,
) -> pd.Series:
    """Compute Moran's I on residuals after regressing out covariates."""
    if genes is None:
        genes = adata.var_names

    x = adata.X
    if issparse(x):
        x = x.toarray()

    gene_indices = [adata.var_names.get_loc(g) for g in genes]
    y = x[:, gene_indices]

    if covariates is not None:
        reg = LinearRegression().fit(covariates, y)
        residuals = y - reg.predict(covariates)
    else:
        residuals = y

    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=k)

    w = adata.obsp["connectivities"]
    n = adata.n_obs
    w_sum = w.sum()

    results = {}
    for i, gene in enumerate(genes):
        z = residuals[:, i]
        z = z - np.mean(z)

        num = n * (z.T @ w @ z)
        den = w_sum * np.sum(z**2)

        morans_index = num / (den + EPSILON)
        results[gene] = morans_index

    return pd.Series(results)


def compute_circular_regression(
    adata: AnnData,
    genes: Iterable[str],
    covariates: Optional[np.ndarray] = None,
    embedding_key: str = "X_umap",
    vantage_point: Optional[tuple | list | np.ndarray | int] = None,
) -> pd.Series:
    """Fit GLM: E ~ sin(theta) + cos(theta) + Z, LRT vs E ~ Z.
    Returns p-value of the F-test."""
    if vantage_point is None:
        if "biorsp" in adata.uns and "params" in adata.uns["biorsp"]:
            vantage_point = adata.uns["biorsp"]["params"]["vantage_point"]
        else:
            vantage_point = np.mean(adata.obsm[embedding_key], axis=0)

    vpoint = np.asarray(vantage_point)
    _, theta = cartesian_to_polar(adata.obsm[embedding_key], vpoint)

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    if covariates is None:
        z = np.zeros((adata.n_obs, 0))
    else:
        z = np.asarray(covariates)

    x_full = np.column_stack([sin_t, cos_t, z])
    x_null = z

    if x_null.shape[1] == 0:
        x_null = np.zeros((adata.n_obs, 0))

    y = adata[:, genes].X
    if issparse(y):
        y = y.toarray()

    reg_full = LinearRegression().fit(x_full, y)
    rss_full = np.sum((y - reg_full.predict(x_full)) ** 2, axis=0)
    df_full = adata.n_obs - x_full.shape[1] - 1

    if x_null.shape[1] > 0:
        reg_null = LinearRegression().fit(x_null, y)
        rss_null = np.sum((y - reg_null.predict(x_null)) ** 2, axis=0)
        df_null = adata.n_obs - x_null.shape[1] - 1
    else:
        rss_null = np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0)
        df_null = adata.n_obs - 1

    num = (rss_null - rss_full) / (df_null - df_full)
    den = rss_full / df_full
    f_stat = num / (den + EPSILON)
    p_values = f.sf(f_stat, df_null - df_full, df_full)

    return pd.Series(p_values, index=genes)


def compute_gradient_magnitude_corrected(
    adata: AnnData,
    genes: Iterable[str],
    covariates: np.ndarray,
    embedding_key: str = "X_umap",
) -> pd.Series:
    """Gradient magnitude of residuals."""
    if genes is None:
        genes = adata.var_names

    x = adata.X
    if issparse(x):
        x = x.toarray()

    gene_indices = [adata.var_names.get_loc(g) for g in genes]
    y = x[:, gene_indices]

    if covariates is not None:
        reg = LinearRegression().fit(covariates, y)
        residuals = y - reg.predict(covariates)
    else:
        residuals = y

    coords = adata.obsm[embedding_key]
    nbrs = NearestNeighbors(n_neighbors=30).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    results = {}
    for i, gene in enumerate(genes):
        expr = residuals[:, i]

        grads = []
        for j in range(adata.n_obs):
            nbr_indices = indices[j]
            local_coords = coords[nbr_indices] - coords[j]
            local_expr = expr[nbr_indices] - expr[j]

            grad, _, _, _ = np.linalg.lstsq(local_coords, local_expr, rcond=None)
            grads.append(np.linalg.norm(grad))

        results[gene] = np.mean(grads)

    return pd.Series(results)
