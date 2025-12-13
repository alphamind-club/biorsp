import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .api import scan_genes, set_vantage


def check_embedding_stability(
    adata, genes, embeddings=["X_umap", "X_pca"], vantage_mode="density_peak", n_perm=0
) -> pd.DataFrame:
    """
    Check stability of results across multiple embeddings.

    Parameters
    ----------
    embeddings : list of str
        Keys in adata.obsm to use.

    Returns
    -------
    stability_report : pd.DataFrame
        Stability metrics for each gene.
    """
    results_by_embedding = {}

    for emb in embeddings:
        if emb not in adata.obsm:
            print(f"Warning: {emb} not found in adata.obsm. Skipping.")
            continue

        print(f"Running on {emb}...")
        try:
            vantage = set_vantage(adata, key=emb, mode=vantage_mode)
        except Exception as e:
            print(f"Failed to set vantage for {emb}: {e}")
            continue

        res: pd.DataFrame = scan_genes(
            adata, genes, vantage_point=vantage, embedding_key=emb, n_perm=n_perm
        )
        results_by_embedding[emb] = res

    if len(results_by_embedding) < 2:
        raise ValueError("Need at least 2 valid embeddings to check stability.")

    stability_rows = []

    emb_keys = list(results_by_embedding.keys())
    a1_corrs = []

    for i in range(len(emb_keys)):
        for j in range(i + 1, len(emb_keys)):
            k1, k2 = emb_keys[i], emb_keys[j]
            df1, df2 = results_by_embedding[k1], results_by_embedding[k2]

            common_genes = df1.index.intersection(df2.index)
            if len(common_genes) < 2:
                continue

            corr, _ = spearmanr(
                df1.loc[common_genes, "A1"], df2.loc[common_genes, "A1"]
            )
            a1_corrs.append(corr)

    global_a1_stability = np.mean(a1_corrs) if a1_corrs else np.nan
    print(f"Global A1 Stability (Spearman Rho): {global_a1_stability:.3f}")

    for gene in genes:
        thetas = []
        a1s = []

        for k in emb_keys:
            if gene in results_by_embedding[k].index:
                thetas.append(results_by_embedding[k].loc[gene, "theta_hat"])
                a1s.append(results_by_embedding[k].loc[gene, "A1"])

        if len(thetas) < 2:
            continue


        z = np.sum(np.exp(1j * np.array(thetas)))
        R = np.abs(z) / len(thetas)
        angular_dispersion = 1 - R

        a1_mean: np.floating[np.Any] = np.mean(a1s)
        a1_std: np.floating[np.Any] = np.std(a1s)
        a1_cv: np.floating[np.Any] = a1_std / (a1_mean + 1e-6)

        stability_rows.append(
            {
                "gene": gene,
                "angular_dispersion": angular_dispersion,
                "a1_cv": a1_cv,
                "mean_a1": a1_mean,
            }
        )

    return pd.DataFrame(stability_rows).set_index("gene")
