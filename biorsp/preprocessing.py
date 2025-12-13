import warnings

import numpy as np
from scipy.sparse import issparse
from sklearn.mixture import GaussianMixture


def get_expression_vector(adata, gene, layer=None):
    """
    Retrieve expression vector for a gene from AnnData.
    """
    if gene not in adata.var_names:
        raise ValueError(f"Gene {gene} not found in adata.")

    idx = adata.var_names.get_loc(gene)

    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if issparse(X):
        expr = X[:, idx].toarray().flatten()
    else:
        expr = X[:, idx].flatten()

    return expr


def check_normalization(adata):
    """
    Heuristic check if data is log-normalized.
    """
    if issparse(adata.X):
        max_val = adata.X.max()
    else:
        max_val = np.max(adata.X)

    if max_val > 20:
        warnings.warn(
            "Max expression value > 20. Data might not be log-normalized. BioRSP expects log-normalized data for best results."
        )


def compute_gene_weights(adata, gene, method="log_normalized", q=0.9, layer=None):
    """
    Compute continuous weights for a gene, replacing hard thresholding.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    gene : str
        Gene name.
    method : str
        'log_normalized': w = x / max(x) (simple scaling)
        'soft_threshold': w = clipped linear ramp around quantile q
        'mixture': w = posterior probability of high component in GMM
    q : float
        Quantile parameter for 'soft_threshold'.
    layer : str, optional
        Layer to use for expression.

    Returns
    -------
    weights : np.ndarray
        Continuous weights in [0, 1].
    """
    check_normalization(adata)
    expr = get_expression_vector(adata, gene, layer=layer)

    if method == "log_normalized":
        max_val = np.quantile(expr, 0.999)
        if max_val == 0:
            max_val = 1.0
        weights = np.clip(expr / max_val, 0, 1)

    elif method == "soft_threshold":
        thresh = np.quantile(expr, q)
        width = max(thresh * 0.2, 0.1)

        lower = thresh - width
        upper = thresh + width

        if upper == lower:
            weights = (expr > lower).astype(float)
        else:
            weights = np.clip((expr - lower) / (upper - lower), 0, 1)

    elif method == "mixture":

        X = expr.reshape(-1, 1)

        if np.sum(expr > 0) < 50:
            max_val = np.max(expr) if np.max(expr) > 0 else 1
            return expr / max_val

        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(X)

            means = gmm.means_.flatten()
            high_idx = np.argmax(means)

            probs = gmm.predict_proba(X)
            weights = probs[:, high_idx]
        except Exception:
            weights = np.clip(expr / np.max(expr), 0, 1)

    else:
        raise ValueError(f"Unknown method: {method}")

    return weights
