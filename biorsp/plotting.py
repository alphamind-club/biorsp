"""Plotting helpers for BioRSP visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

if TYPE_CHECKING:
    from anndata import AnnData


def plot_rsp(
    adata: AnnData,
    gene: str,
    ax: plt.Axes | None = None,
    *,
    show: bool = True,
    ci_lower: np.ndarray | None = None,
    ci_upper: np.ndarray | None = None,
    label: str | None = None,
) -> plt.Axes:
    """Plot the RSP curve for a single gene in polar coordinates.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with ``biorsp`` results present in ``.uns``.
    gene : str
        Gene to plot.
    ax : matplotlib.axes.Axes, optional
        A pre-existing axis to draw onto.
    show : bool
        If True, call ``plt.show()`` after plotting.
    ci_lower, ci_upper : np.ndarray, optional
        Optional confidence intervals for shading the curve.
    label : str, optional
        Custom label for the legend. If None, uses the gene name.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plot."""
    if "biorsp" not in adata.uns or "rsp_curves" not in adata.uns["biorsp"]:
        msg = "Run find_spatially_patterned_genes first to compute RSP curves."
        raise ValueError(msg)

    rsp_df = adata.uns["biorsp"]["rsp_curves"]
    if gene not in rsp_df.columns:
        msg = f"Gene {gene} not found in results."
        raise ValueError(msg)

    angles = rsp_df.index.to_numpy()
    values = rsp_df[gene].to_numpy()

    angles_plot = np.concatenate([angles, [angles[0] + 2 * np.pi]])
    values_plot = np.concatenate([values, [values[0]]])

    if ax is None:
        _fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    else:
        is_polar = False
        try:
            is_polar = (
                getattr(ax, "name", "") == "polar"
                or getattr(ax, "projection", "") == "polar"
            )
        except Exception:
            is_polar = False
        if not is_polar:
            fig = ax.figure
            pos = ax.get_position()
            try:
                ax.remove()
            except Exception:
                pass
            ax = fig.add_axes(pos, projection="polar")

    plot_label = label if label is not None else gene
    ax.plot(angles_plot, values_plot, label=plot_label, linewidth=2)

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
        angles_plot,
        0,
        values_plot,
        where=(values_plot > 0),
        alpha=0.3,
        color="red",
    )
    ax.fill_between(
        angles_plot,
        0,
        values_plot,
        where=(values_plot < 0),
        alpha=0.3,
        color="blue",
    )

    title = f"RSP: {plot_label}"
    if gene in adata.var_names:
        if "ARIA" in adata.var.columns:
            cra = adata.var.loc[gene, "ARIA"]
            title += f"\nARIA={cra:.3f}"
        if "A1" in adata.var.columns:
            a1 = adata.var.loc[gene, "A1"]
            title += f", A1={a1:.2f}"
        if "A2" in adata.var.columns:
            a2 = adata.var.loc[gene, "A2"]
            title += f", A2={a2:.2f}"
        if "p_A1" in adata.var.columns:
            p = adata.var.loc[gene, "p_A1"]
            title += f", p={p:.3f}"

    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    has_labels = any(lbl and not str(lbl).startswith("_") for lbl in labels)
    if has_labels:
        ax.legend()

    if show:
        plt.show()

    return ax


def plot_embedding_with_sectors(
    adata: AnnData,
    gene: str,
    vantage_point: tuple | list | np.ndarray | int | None = None,
    embedding_key: str = "X_umap",
    ax: plt.Axes | None = None,
    *,
    show: bool = True,
) -> plt.Axes:
    """Plot embedding with vantage point and preferred direction arrow for a gene."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 8))

    sc.pl.embedding(
        adata,
        basis=embedding_key,
        color=gene,
        ax=ax,
        show=False,
        frameon=False,
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
                coords[0],
                coords[1],
                c="red",
                marker="x",
                s=100,
                label="Vantage",
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

    handles, labels = ax.get_legend_handles_labels()
    has_labels = any(lbl and not str(lbl).startswith("_") for lbl in labels)
    if has_labels:
        ax.legend()

    if show:
        plt.show()

    return ax
