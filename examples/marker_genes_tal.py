import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

import biorsp

SIGNIFICANCE_ALPHA = 0.05
TOP_GENES = 20
PREVIEW_TOP = 5
NORMALIZE_THRESHOLD = 20

summary_stats = []

adata = sc.read_h5ad("data/kpmp_sn.h5ad")

tal_mask = adata.obs["subclass.l1"] == "TAL"

diseases = ["normal", "chronic kidney disease", "acute kidney failure"]
for disease in diseases:
    disease_mask = adata.obs["disease"] == disease
    disease_slug = disease.replace(" ", "_")
    adata_tal = adata[tal_mask & disease_mask].copy()
    if adata_tal.n_obs == 0:
        continue

    if "feature_name" in adata_tal.var.columns:
        adata_tal.var.index = adata_tal.var["feature_name"].astype(str)
        adata_tal.var_names_make_unique()

    if adata_tal.X.max() > NORMALIZE_THRESHOLD:
        sc.pp.normalize_total(adata_tal, target_sum=1e4)
        sc.pp.log1p(adata_tal)

    adata_tal.obsm["X_spatial"] = adata_tal.obsm["X_umap"]

    vantage = biorsp.define_reference_point(adata_tal, method="geometric_median")

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

    results = biorsp.find_spatially_patterned_genes(
        adata_tal,
        genes_to_test=target_genes,
        reference_point=vantage,
        coordinate_system="X_umap",
        num_permutations=100,
        permutation_method="knn",
    )

    sig_genes = results[results["p_CRA"] < SIGNIFICANCE_ALPHA].sort_values(
        "ARIA", ascending=False
    )

    summary_stats.append(
        {
            "disease": disease,
            "n_sig": len(sig_genes),
            "top_cra": sig_genes["ARIA"].max() if len(sig_genes) > 0 else 0,
        }
    )

    if len(sig_genes) > 1:
        top_genes = sig_genes.head(TOP_GENES).index.tolist()

        rsp_curves = adata_tal.uns["biorsp"]["rsp_curves"][top_genes]

        corr_matrix = rsp_curves.corr()

        import os

        os.makedirs("examples/outputs", exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
        grid = rsp_curves.index.to_numpy()
        angles_plot = np.concatenate([grid, [grid[0] + 2 * np.pi]])

        for gene in top_genes[:PREVIEW_TOP]:
            vals = rsp_curves[gene].to_numpy()
            vals_plot = np.concatenate([vals, [vals[0]]])
            ax.plot(angles_plot, vals_plot, label=gene, linewidth=2)

        ax.set_title("RSP Curves of Top TAL Directional Markers (UMAP Space)")
        ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0))
        plt.savefig(
            f"examples/outputs/tal_{disease_slug}_top_markers_rsp.png",
            bbox_inches="tight",
        )

        g = sns.clustermap(
            corr_matrix,
            cmap="coolwarm",
            center=0,
            annot=False,
            xticklabels=True,
            yticklabels=True,
            figsize=(12, 10),
        )
        g.savefig(
            f"examples/outputs/tal_{disease_slug}_gene_correlation.png",
            bbox_inches="tight",
        )

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
            vals = rsp_curves[gene].to_numpy()
            vals_plot = np.concatenate([vals, [vals[0]]])

            ax2.plot(angles_plot, vals_plot, linewidth=2, color="red")
            ax2.fill_between(angles_plot, 0, vals_plot, alpha=0.1, color="red")

            if "theta_dir" in sig_genes.columns:
                theta_hat = sig_genes.loc[gene, "theta_dir"]
                ax2.annotate(
                    "",
                    xy=(theta_hat, np.max(vals_plot)),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2),
                )

            ax2.set_title(
                f"{gene} RSP Curve\n(Directional Deviance: {sig_genes.loc[gene, 'D_dir']:.2f})",
                fontsize=14,
            )

            plt.suptitle(f"Spatial Analysis: {gene}", fontsize=18)
            plt.tight_layout()
            plt.savefig(
                f"examples/outputs/tal_{disease_slug}_analysis_{gene}.png",
                bbox_inches="tight",
            )
            plt.close(fig)

        sig_genes.to_csv(f"examples/outputs/tal_{disease_slug}_significant_genes.csv")

if summary_stats:
    df_summary = pd.DataFrame(summary_stats)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(df_summary["disease"], df_summary["n_sig"], color="skyblue")
    ax.set_title("Number of Significant Spatially Patterned Genes per Disease")
    ax.set_xlabel("Disease")
    ax.set_ylabel("Number of Significant Genes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("examples/outputs/summary_significant_genes.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_summary["n_sig"], df_summary["top_cra"], s=100)
    for i, row in df_summary.iterrows():
        ax.annotate(
            row["disease"],
            (row["n_sig"], row["top_cra"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_title("Top ARIA vs Number of Significant Genes")
    ax.set_xlabel("Number of Significant Genes")
    ax.set_ylabel("Top ARIA Value")
    plt.tight_layout()
    plt.savefig("examples/outputs/summary_cra_vs_sig.png")
    plt.close()
