import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import biorsp


print("Loading data...")
adata = sc.read_h5ad("data/kpmp_sn.h5ad")

print("Subsetting for TAL cells...")
tal_mask = adata.obs["subclass.l1"] == "TAL"
adata_tal = adata[tal_mask].copy()

print(f"TAL cells found: {adata_tal.n_obs}")

if "feature_name" in adata_tal.var.columns:
    adata_tal.var.index = adata_tal.var["feature_name"].astype(str)
    adata_tal.var_names_make_unique()

if adata_tal.X.max() > 20:
    print("Data appears to be raw counts. Normalizing...")
    sc.pp.normalize_total(adata_tal, target_sum=1e4)
    sc.pp.log1p(adata_tal)

adata_tal.obsm["X_spatial"] = adata_tal.obsm["X_umap"]

print("Setting up BioRSP...")
vantage = biorsp.set_vantage(adata_tal, mode="geometric_median")

tal_marker_genes = [
    "SLC12A1",
    "PROM1",
    "DCDC2",
    "ITGB6",
    "EGF",
    "CDH11",
    "ESRRB",
    "DCDC1",
    "HAVCR1",
    "SPP1",
    "VCAM1",
    "CST3",
    "CLU",
    "IGFBP7",
    "DEFB1",
    "EGFR",
    "NRG1",
    "NRG3",
    "ERBB4",
    "ERBB2",
    "ITGB8",
    "ITGAV",
    "ESRRG",
    "ESRRA",
    "SMAD2",
    "SMAD3",
    "STAT1",
    "STAT3",
    "STAT5B",
    "CCL5",
    "CCL21",
    "FGF2",
    "FGF13",
    "FGFR1",
    "FGFR2",
    "COL1A1",
    "ACTA2",
    "JUN",
    "FOS",
    "NFKB1",
    "REL",
    "RELB",
    "SKIL",
    "BICC1",
]

target_genes = [gene for gene in tal_marker_genes if gene in adata_tal.var_names]
print(
    f"Scanning {len(target_genes)} TAL marker genes out of {len(tal_marker_genes)} provided..."
)

print("Running BioRSP scan (this may take a moment)...")
results = biorsp.scan_genes(adata_tal, target_genes, vantage, n_perm=100)

sig_genes = results[results["p_value"] < 0.05].sort_values("A1", ascending=False)
print(f"Found {len(sig_genes)} significant directional genes.")
print(sig_genes.head(10))

if len(sig_genes) > 1:
    print("Analyzing gene-gene relationships...")
    top_genes = sig_genes.head(20).index.tolist()

    rsp_curves = adata_tal.uns["biorsp"]["rsp_curves"][top_genes]

    corr_matrix = rsp_curves.corr()

    import os

    os.makedirs("analysis_results", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    grid = rsp_curves.index.values
    angles_plot = np.concatenate([grid, [grid[0] + 2 * np.pi]])

    for gene in top_genes[:5]:  # Plot top 5
        vals = rsp_curves[gene].values
        vals_plot = np.concatenate([vals, [vals[0]]])
        ax.plot(angles_plot, vals_plot, label=gene, linewidth=2)

    ax.set_title("RSP Curves of Top TAL Directional Markers (UMAP Space)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0))
    plt.savefig("analysis_results/tal_top_markers_rsp.png", bbox_inches="tight")
    print("Saved analysis_results/tal_top_markers_rsp.png")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        annot=False,
        xticklabels=True,
        yticklabels=True,
    )
    plt.title("Gene-Gene Spatial Correlation (RSP Similarity)")
    plt.savefig("analysis_results/tal_gene_correlation.png", bbox_inches="tight")
    print("Saved analysis_results/tal_gene_correlation.png")

    print(f"Generating side-by-side plots for {len(top_genes)} top genes...")

    for gene in top_genes:
        fig = plt.figure(figsize=(16, 7))

        ax1 = fig.add_subplot(121)
        x, y = adata_tal.obsm["X_umap"][:, 0], adata_tal.obsm["X_umap"][:, 1]
        c = (
            adata_tal[:, gene].X.toarray().flatten()
            if hasattr(adata_tal.X, "toarray")
            else adata_tal[:, gene].X.flatten()
        )

        order = np.argsort(c)
        x_sorted, y_sorted, c_sorted = x[order], y[order], c[order]

        mask_zero = c_sorted == 0
        mask_expr = c_sorted > 0

        ax1.scatter(
            x_sorted[mask_zero],
            y_sorted[mask_zero],
            c="lightgray",
            s=10,
            alpha=0.5,
            label="No Expression",
        )

        im = ax1.scatter(
            x_sorted[mask_expr],
            y_sorted[mask_expr],
            c=c_sorted[mask_expr],
            cmap="Reds",
            s=10,
            alpha=0.8,
        )

        ax1.scatter(
            vantage[0],
            vantage[1],
            c="blue",
            marker="*",
            s=200,
            label="Vantage Point",
            edgecolors="white",
        )
        ax1.set_title(f"{gene} Expression (UMAP)", fontsize=14)
        plt.colorbar(im, ax=ax1, label="Expression")
        ax1.legend()
        ax1.axis("off")

        ax2 = fig.add_subplot(122, projection="polar")
        vals = rsp_curves[gene].values
        vals_plot = np.concatenate([vals, [vals[0]]])

        ax2.plot(angles_plot, vals_plot, linewidth=2, color="red")
        ax2.fill_between(angles_plot, 0, vals_plot, alpha=0.1, color="red")

        theta_hat = sig_genes.loc[gene, "theta_hat"]
        ax2.annotate(
            "",
            xy=(theta_hat, np.max(vals_plot)),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="black", lw=2),
        )

        ax2.set_title(
            f"{gene} RSP Curve\n(A1={sig_genes.loc[gene, 'A1']:.2f})", fontsize=14
        )

        plt.suptitle(f"Spatial Analysis: {gene}", fontsize=18)
        plt.tight_layout()
        plt.savefig(f"analysis_results/tal_analysis_{gene}.png", bbox_inches="tight")
        plt.close(fig)

    print("Saved individual gene analysis plots.")

    sig_genes.to_csv("analysis_results/tal_significant_genes.csv")
    print("Saved analysis_results/tal_significant_genes.csv")

else:
    print("No significant genes found.")
