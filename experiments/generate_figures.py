import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import biorsp
from experiments.synthetic_data import create_synthetic_dataset, add_gene_expression
from biorsp.baselines import run_baselines

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)


def save_fig(fig, name):
    fig.savefig(f"output/figures/{name}.png", dpi=300, bbox_inches="tight")
    print(f"Saved output/figures/{name}.png")


def plot_confounder_demo():
    print("Generating Confounder Demo Figure...")
    adata = create_synthetic_dataset(n_cells=1000, manifold="blobs", n_genes=10)
    vantage = biorsp.set_vantage(adata, mode="geometric_median")
    target_gene = "Gene_5"  # Cluster marker

    res_global = biorsp.scan_genes(
        adata, [target_gene], vantage, n_perm=100, stratify_key=None
    )
    res_strat = biorsp.scan_genes(
        adata, [target_gene], vantage, n_perm=100, stratify_key="clusters"
    )


    from biorsp.metrics import compute_rsp_curve, run_stratified_permutation
    from biorsp.geometry import cartesian_to_polar, define_angular_grid
    from biorsp.preprocessing import compute_gene_weights

    coords = adata.obsm["X_umap"]
    r, theta = cartesian_to_polar(coords, vantage)
    weights = compute_gene_weights(adata, target_gene)
    bg_weights = np.ones(len(r))
    grid = define_angular_grid(5)

    obs = compute_rsp_curve(
        theta, r, weights, bg_weights, grid, window_width_deg=30, r_min=None, r_max=None
    )
    rsp_obs = obs["rsp"]

    null_global = run_stratified_permutation(
        theta,
        r,
        weights,
        bg_weights,
        grid,
        window_width_deg=30,
        r_min=None,
        r_max=None,
        stratify_by=None,
        n_perm=100,
    )
    null_rsp_global = null_global["null_rsp"]  # (n_perm, n_grid)

    stratify_by = adata.obs["clusters"].values
    null_strat = run_stratified_permutation(
        theta,
        r,
        weights,
        bg_weights,
        grid,
        window_width_deg=30,
        r_min=None,
        r_max=None,
        stratify_by=stratify_by,
        n_perm=100,
    )
    null_rsp_strat = null_strat["null_rsp"]

    fig = plt.figure(figsize=(18, 6))

    angles = grid
    angles_plot = np.concatenate([angles, [angles[0] + 2 * np.pi]])

    def plot_band(ax, null_matrix, color, label):
        lower = np.percentile(null_matrix, 2.5, axis=0)
        upper = np.percentile(null_matrix, 97.5, axis=0)
        mean = np.mean(null_matrix, axis=0)

        l_plot = np.concatenate([lower, [lower[0]]])
        u_plot = np.concatenate([upper, [upper[0]]])
        m_plot = np.concatenate([mean, [mean[0]]])

        ax.fill_between(
            angles_plot, l_plot, u_plot, color=color, alpha=0.3, label=f"{label} 95% CI"
        )
        ax.plot(angles_plot, m_plot, color=color, linestyle="--", alpha=0.7)

    ax1 = fig.add_subplot(131)
    x, y = coords[:, 0], coords[:, 1]
    c = adata[:, target_gene].X.flatten()
    im = ax1.scatter(x, y, c=c, cmap="viridis", s=20, alpha=0.8)
    ax1.set_title(f"{target_gene} Expression\n(Cluster Marker)", fontsize=14)
    ax1.axis("off")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax1.scatter(
        vantage[0], vantage[1], c="red", marker="*", s=200, label="Vantage Point"
    )
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(132, projection="polar")
    rsp_plot = np.concatenate([rsp_obs, [rsp_obs[0]]])
    ax2.plot(angles_plot, rsp_plot, "k-", linewidth=2, label="Observed RSP")
    plot_band(ax2, null_rsp_global, "red", "Global Null")
    ax2.set_title(
        f"Naive Permutation\n(False Positive: p={res_global.loc[target_gene, 'p_value']:.3f})",
        fontsize=14,
    )
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25))

    ax3 = fig.add_subplot(133, projection="polar")
    ax3.plot(angles_plot, rsp_plot, "k-", linewidth=2, label="Observed RSP")
    plot_band(ax3, null_rsp_strat, "green", "Stratified Null")
    ax3.set_title(
        f"Stratified Permutation\n(True Negative: p={res_strat.loc[target_gene, 'p_value']:.3f})",
        fontsize=14,
    )
    ax3.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25))

    plt.suptitle(f"Confounder Control: {target_gene}", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "confounder_demo")


def plot_condition_shift():
    print("Generating Condition Shift Figure...")
    adata = create_synthetic_dataset(n_cells=1000, manifold="circle")
    vantage = biorsp.set_vantage(adata, mode="geometric_median")
    target_gene = "Gene_0"

    res_ctrl = biorsp.scan_genes(adata, [target_gene], vantage, n_perm=0)
    rsp_ctrl = adata.uns["biorsp"]["rsp_curves"][target_gene].values

    adata_disease = adata.copy()
    coords = adata.obsm["X_umap"]
    t = adata.obs["latent_t"].values

    theta_rot = -np.pi / 2
    rot_mat = np.array(
        [
            [np.cos(theta_rot), -np.sin(theta_rot)],
            [np.sin(theta_rot), np.cos(theta_rot)],
        ]
    )
    coords_rot = coords @ rot_mat.T

    counts_disease = add_gene_expression(
        adata_disease,
        coords_rot,
        t,
        "directional",
        vantage_point=vantage,
        effect_size=2.5,
    )
    adata_disease.X[:, adata.var_names.get_loc(target_gene)] = counts_disease

    res_dis = biorsp.scan_genes(adata_disease, [target_gene], vantage, n_perm=0)
    rsp_dis = adata_disease.uns["biorsp"]["rsp_curves"][target_gene].values

    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(131)
    x, y = coords[:, 0], coords[:, 1]
    c_ctrl = adata[:, target_gene].X.flatten()
    im1 = ax1.scatter(x, y, c=c_ctrl, cmap="viridis", s=20, alpha=0.8)
    ax1.set_title("Control Condition", fontsize=14)
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.scatter(vantage[0], vantage[1], c="red", marker="*", s=200)

    ax2 = fig.add_subplot(132)
    c_dis = adata_disease[:, target_gene].X.flatten()
    im2 = ax2.scatter(x, y, c=c_dis, cmap="viridis", s=20, alpha=0.8)
    ax2.set_title("Disease Condition\n(Rotated Expression)", fontsize=14)
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.scatter(vantage[0], vantage[1], c="red", marker="*", s=200)

    ax3 = fig.add_subplot(133, projection="polar")

    grid = adata.uns["biorsp"]["rsp_curves"].index.values
    angles_plot = np.concatenate([grid, [grid[0] + 2 * np.pi]])

    vals_c = np.concatenate([rsp_ctrl, [rsp_ctrl[0]]])
    ax3.plot(angles_plot, vals_c, "b-", linewidth=2, label="Control")
    ax3.fill_between(angles_plot, 0, vals_c, alpha=0.1, color="b")

    vals_d = np.concatenate([rsp_dis, [rsp_dis[0]]])
    ax3.plot(angles_plot, vals_d, "r-", linewidth=2, label="Disease")
    ax3.fill_between(angles_plot, 0, vals_d, alpha=0.1, color="r")

    theta_c = res_ctrl.loc[target_gene, "theta_hat"]
    theta_d = res_dis.loc[target_gene, "theta_hat"]

    ax3.annotate(
        "",
        xy=(theta_c, np.max(vals_c)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="b", lw=2),
    )
    ax3.annotate(
        "",
        xy=(theta_d, np.max(vals_d)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="r", lw=2),
    )

    ax3.set_title(f"RSP Shift: {target_gene}", fontsize=14)
    ax3.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25))

    plt.suptitle("Condition Shift Analysis", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "condition_shift")


def plot_baselines_comparison():
    print("Generating Baselines Comparison Figure...")
    adata = create_synthetic_dataset(n_cells=1000, manifold="blobs", n_genes=20)
    vantage = biorsp.set_vantage(adata, mode="geometric_median")

    res_bio = biorsp.scan_genes(
        adata, adata.var_names, vantage, n_perm=50, stratify_key="clusters"
    )

    res_base = run_baselines(adata, adata.var_names)

    df = pd.concat([res_bio, res_base], axis=1)
    df["Type"] = adata.var["ground_truth"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(
        data=df, x="Morans_I", y="A1", hue="Type", style="Type", s=100, ax=axes[0]
    )
    axes[0].set_title("BioRSP A1 vs Moran's I")
    axes[0].axhline(y=0.1, color="k", linestyle="--", alpha=0.5)
    axes[0].text(0.1, 0.05, "BioRSP Null Zone", fontsize=10)

    df["neg_log_p"] = -np.log10(df["p_value"] + 1e-3)
    sns.scatterplot(
        data=df,
        x="Morans_I",
        y="neg_log_p",
        hue="Type",
        style="Type",
        s=100,
        ax=axes[1],
    )
    axes[1].set_title("BioRSP Significance vs Moran's I")
    axes[1].axhline(y=-np.log10(0.05), color="r", linestyle="--", label="p=0.05")

    plt.suptitle(
        "Baseline Comparison: Separating Directional from Clustered", fontsize=16
    )
    save_fig(fig, "baseline_comparison")


def plot_scalability():
    print("Generating Scalability Figure...")
    import time

    sizes = [1000, 5000, 10000, 20000]
    times = []

    for n in sizes:
        adata = create_synthetic_dataset(n_cells=n, n_genes=5)
        vantage = biorsp.set_vantage(adata, mode="geometric_median")
        start = time.time()
        biorsp.scan_genes(adata, ["Gene_0"], vantage, n_perm=50)
        times.append(time.time() - start)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sizes, times, "o-", linewidth=2)
    ax.set_xlabel("Number of Cells")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("BioRSP Scalability (50 permutations)")

    z = np.polyfit(sizes, times, 1)
    p = np.poly1d(z)
    ax.plot(sizes, p(sizes), "r--", alpha=0.5, label=f"Linear Fit (slope={z[0]:.2e})")
    ax.legend()

    save_fig(fig, "scalability")


def main():
    import os

    os.makedirs("figures", exist_ok=True)

    plot_confounder_demo()
    plot_condition_shift()
    plot_baselines_comparison()
    plot_scalability()

    print("All figures generated in output/figures/")


if __name__ == "__main__":
    main()
