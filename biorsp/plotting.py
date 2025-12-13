import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


def plot_rsp(adata, gene, ax=None, show=True, ci_lower=None, ci_upper=None):
    """
    Plot RSP curve for a gene in polar coordinates.

    Parameters
    ----------
    ci_lower : np.ndarray, optional
        Lower bound of confidence interval (same length as angles).
    ci_upper : np.ndarray, optional
        Upper bound of confidence interval.
    """
    if "biorsp" not in adata.uns or "rsp_curves" not in adata.uns["biorsp"]:
        raise ValueError("Run scan_genes first to compute RSP curves.")

    rsp_df = adata.uns["biorsp"]["rsp_curves"]
    if gene not in rsp_df.columns:
        raise ValueError(f"Gene {gene} not found in results.")

    angles = rsp_df.index.values
    values = rsp_df[gene].values

    angles_plot = np.concatenate([angles, [angles[0] + 2 * np.pi]])
    values_plot = np.concatenate([values, [values[0]]])

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    ax.plot(angles_plot, values_plot, label=gene, linewidth=2)

    if ci_lower is not None and ci_upper is not None:
        ci_lower_plot = np.concatenate([ci_lower, [ci_lower[0]]])
        ci_upper_plot = np.concatenate([ci_upper, [ci_upper[0]]])
        ax.fill_between(
            angles_plot,
            ci_lower_plot,
            ci_upper_plot,
            color="gray",
            alpha=0.2,
            label="95% CI",
        )

    ax.plot(angles_plot, np.zeros_like(angles_plot), "k--", alpha=0.5)

    ax.fill_between(
        angles_plot, 0, values_plot, where=(values_plot > 0), alpha=0.3, color="red"
    )
    ax.fill_between(
        angles_plot, 0, values_plot, where=(values_plot < 0), alpha=0.3, color="blue"
    )

    title = f"RSP: {gene}"
    if gene in adata.var_names:
        if "A1" in adata.var.columns:
            a1 = adata.var.loc[gene, "A1"]
            title += f"\nA1={a1:.2f}"
        if "A2" in adata.var.columns:
            a2 = adata.var.loc[gene, "A2"]
            title += f", A2={a2:.2f}"
        if "p_A1" in adata.var.columns:
            p = adata.var.loc[gene, "p_A1"]
            title += f", p={p:.3f}"

    ax.set_title(title)

    if show:
        plt.show()

    return ax


def plot_embedding_with_sectors(
    adata, gene, vantage_point=None, embedding_key="X_umap", ax=None, show=True
):
    """
    Plot embedding with vantage point and preferred direction.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    sc.pl.embedding(
        adata, basis=embedding_key, color=gene, ax=ax, show=False, frameon=False
    )

    if vantage_point is None:
        if "biorsp" in adata.uns and "params" in adata.uns["biorsp"]:
            vantage_point = adata.uns["biorsp"]["params"]["vantage_point"]
        else:
            pass

    if vantage_point is not None:
        if isinstance(vantage_point, (int, np.integer)):
            coords = adata.obsm[embedding_key][vantage_point]
            ax.scatter(
                coords[0], coords[1], c="red", marker="x", s=100, label="Vantage"
            )
        else:
            ax.scatter(
                vantage_point[0],
                vantage_point[1],
                c="red",
                marker="x",
                s=100,
                label="Vantage",
            )

    if gene in adata.var_names and "theta_hat" in adata.var.columns:
        theta = adata.var.loc[gene, "theta_hat"]
        xlim = ax.get_xlim()
        scale = (xlim[1] - xlim[0]) * 0.2

        dx = scale * np.cos(theta)
        dy = scale * np.sin(theta)

        if vantage_point is not None:
            if isinstance(vantage_point, (int, np.integer)):
                start = adata.obsm[embedding_key][vantage_point]
            else:
                start = vantage_point

            ax.arrow(
                start[0],
                start[1],
                dx,
                dy,
                head_width=scale * 0.2,
                head_length=scale * 0.2,
                fc="k",
                ec="k",
                label="Preferred Dir",
            )

    ax.legend()

    if show:
        plt.show()

    return ax


def plot_diagnostics(adata, ax=None, show=True):
    """
    Plot A1 vs A2 distribution.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    if "A1" in adata.var.columns and "A2" in adata.var.columns:
        ax.scatter(adata.var["A1"], adata.var["A2"], alpha=0.5)
        ax.set_xlabel("A1 (Coverage Bias)")
        ax.set_ylabel("A2 (Angular Skew)")
        ax.set_title("Global Directionality Landscape")

    if show:
        plt.show()

    return ax
