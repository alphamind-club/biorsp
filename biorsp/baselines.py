import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors


def compute_morans_i(adata, genes=None, embedding_key="X_umap", k=30):
    """
    Compute Moran's I for spatial autocorrelation on the embedding graph.
    """
    if genes is None:
        genes = adata.var_names

    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=k)


    try:
        morans = sc.metrics.morans_i(adata)
        if isinstance(morans, np.ndarray):
            morans: pd.Series[float] = pd.Series(morans, index=adata.var_names)
        return morans[genes] if genes is not None else morans
    except AttributeError:
        return _morans_i_custom(adata, genes, k)


def _morans_i_custom(adata, genes, k) -> pd.Series[pd.Timestamp]:
    """
    Simple Moran's I implementation.
    """
    W = adata.obsp["connectivities"]

    results = {}
    X = adata.X
    if issparse(X):
        X = X.toarray()

    N = adata.n_obs
    W_sum = W.sum()

    for gene in genes:
        idx = adata.var_names.get_loc(gene)
        x = X[:, idx]
        x_bar = np.mean(x)
        z = x - x_bar


        num = N * (z.T @ W @ z)
        den = W_sum * np.sum(z**2)

        morans_index = num / (den + 1e-9)
        results[gene] = morans_index

    return pd.Series(results)


def compute_gradient_magnitude(
    adata, genes, embedding_key="X_umap", sigma=1.0
) -> pd.Series:
    """
    Estimate gradient magnitude of expression on the embedding using kernel smoothing.
    """
    coords = adata.obsm[embedding_key]
    X = adata.X
    if issparse(X):
        X = X.toarray()

    results = {}


    nbrs: NearestNeighbors = NearestNeighbors(n_neighbors=30).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    for gene in genes:
        idx = adata.var_names.get_loc(gene)
        expr = X[:, idx]

        grads = []
        for i in range(adata.n_obs):
            nbr_indices = indices[i]

            local_coords = coords[nbr_indices] - coords[i]
            local_expr = expr[nbr_indices] - expr[i]

            grad, _, _, _ = np.linalg.lstsq(local_coords, local_expr, rcond=None)
            grads.append(np.linalg.norm(grad))

        results[gene] = np.mean(grads)  # Average gradient magnitude across manifold

    return pd.Series(results)


def compute_neighborhood_enrichment(
    adata, genes, embedding_key="X_umap", k=30
) -> pd.Series:
    """
    Compute enrichment of expression in local neighborhoods vs global background.
    Returns the maximum enrichment score observed across all neighborhoods.
    """
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=k)

    connectivities = adata.obsp["connectivities"]
    adj = (connectivities > 0).astype(float)

    X = adata.X
    if issparse(X):
        X = X.toarray()

    results = {}

    for gene in genes:
        idx = adata.var_names.get_loc(gene)
        expr = X[:, idx]

        degrees = np.array(adj.sum(axis=1)).flatten()
        local_means = (adj @ expr) / (degrees + 1e-9)

        global_mean = np.mean(expr)

        enrichment = np.max(local_means) / (global_mean + 1e-9)
        results[gene] = enrichment

    return pd.Series(results)


def run_baselines(adata, genes, embedding_key="X_umap") -> pd.DataFrame:
    """
    Run all baselines and return a DataFrame.
    """
    print("Computing Moran's I...")
    morans = compute_morans_i(adata, genes, embedding_key)

    print("Computing Gradient Magnitude...")
    grads: pd.Series[pd.Timestamp] = compute_gradient_magnitude(
        adata, genes, embedding_key
    )

    print("Computing Neighborhood Enrichment...")
    enrich: pd.Series[pd.Timestamp] = compute_neighborhood_enrichment(
        adata, genes, embedding_key
    )

    df = pd.DataFrame({"Morans_I": morans, "Gradient_Mag": grads, "Nb_Enrich": enrich})

    return df
