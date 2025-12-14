"""Preprocessing helper utilities used by the BioRSP package."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import issparse
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOG_NORM_THRESHOLD = 20
MIN_NONZERO_FOR_GMM = 50

if TYPE_CHECKING:
    from anndata import AnnData


def get_expression_vector(
    adata: AnnData,
    gene: str,
    layer: str | None = None,
) -> np.ndarray:
    """Retrieve expression vector for a gene from :class:`anndata.AnnData`.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing expression matrix and metadata.
    gene : str
        Gene name to extract.
    layer : str | None
        Optional layer name to use instead of ``adata.X``.

    Returns
    -------
    np.ndarray
        1D array of gene expression values.

    """
    if gene not in adata.var_names:
        msg = f"Gene {gene} not found in adata."
        raise ValueError(msg)

    idx = adata.var_names.get_loc(gene)

    mat = adata.layers[layer] if layer is not None else adata.X

    return mat[:, idx].toarray().flatten() if issparse(mat) else mat[:, idx].flatten()


def check_normalization(adata: AnnData) -> None:
    """Heuristic check whether the expression matrix is log-normalized.

    This function warns if maximum expression values indicate integer-counts
    (not log-scaled), which may affect downstream methods assuming log
    normalization.
    """
    max_val = adata.X.max() if issparse(adata.X) else np.max(adata.X)

    if max_val > LOG_NORM_THRESHOLD:
        msg = (
            "Max expression value > 20. Data might not be log-normalized. "
            "BioRSP expects log-normalized data for best results."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)


def compute_gene_weights(
    adata: AnnData,
    gene: str,
    method: str = "log_normalized",
    q: float = 0.9,
    layer: str | None = None,
) -> np.ndarray:
    """Compute continuous weights for a gene, replacing hard thresholding.

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

        mat = expr.reshape(-1, 1)

        if np.sum(expr > 0) < MIN_NONZERO_FOR_GMM:
            max_val = np.max(expr) if np.max(expr) > 0 else 1
            return expr / max_val

        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(mat)

            means = gmm.means_.flatten()
            high_idx = np.argmax(means)

            probs = gmm.predict_proba(mat)
            weights = probs[:, high_idx]
        except (ValueError, RuntimeError):
            msg = "GMM fit failed; using simple scaling fallback."
            warnings.warn(msg, UserWarning, stacklevel=2)
            weights = np.clip(expr / np.max(expr), 0, 1)

    else:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)

    return weights


def build_covariate_matrix(
    adata: AnnData,
    keys: list[str] | None = None,
    *,
    include_log_depth: bool = True,
) -> np.ndarray:
    """Build covariate matrix ``Z`` for conditional permutation testing.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with ``obs`` and ``layers``.
    keys : Optional[list[str]]
        Observation keys to include. Categorical variables are one-hot encoded.
    include_log_depth : bool
        Whether to include log total counts per cell as a numeric covariate.

    Returns
    -------
    np.ndarray
        Covariate matrix of shape ``(n_obs, n_covariates)``.

    """
    cols = []
    if include_log_depth and "n_counts" in adata.obs:
        depth = np.log1p(adata.obs["n_counts"].to_numpy().astype(float))
        cols.append(depth.reshape(-1, 1))

    if keys:
        cat_cols = []
        num_cols = []
        for k in keys:
            if k not in adata.obs:
                msg = f"Covariate {k} not found in adata.obs"
                raise ValueError(msg)
            if adata.obs[k].dtype.name == "category" or adata.obs[k].dtype == object:
                cat_cols.append(adata.obs[k].astype(str).to_numpy().reshape(-1, 1))
            else:
                num_cols.append(adata.obs[k].to_numpy().astype(float).reshape(-1, 1))

        if cat_cols:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            cols.append(enc.fit_transform(np.concatenate(cat_cols, axis=1)))
        if num_cols:
            num = np.concatenate(num_cols, axis=1)
            cols.append(StandardScaler().fit_transform(num))

    if not cols:
        return np.zeros((adata.n_obs, 1))

    return np.concatenate(cols, axis=1)
