import biorsp
from experiments.synthetic_data import create_synthetic_dataset


def run_confounder_test():
    print("Running Confounder Stress Test...")

    adata = create_synthetic_dataset(n_cells=1000, manifold="blobs", n_genes=20)
    vantage = biorsp.set_vantage(adata, mode="geometric_median")

    target_gene = "Gene_5"  # Cluster marker

    print("  Testing with Global Permutation (Naive)...")
    res_global = biorsp.scan_genes(
        adata, [target_gene], vantage, n_perm=100, stratify_key=None
    )
    p_global = res_global.loc[target_gene, "p_value"]

    print("  Testing with Stratified Permutation (Cluster-Aware)...")
    res_strat = biorsp.scan_genes(
        adata, [target_gene], vantage, n_perm=100, stratify_key="clusters"
    )
    p_strat = res_strat.loc[target_gene, "p_value"]

    print("\nResults:")
    print(f"Gene: {target_gene} (Cluster Marker)")
    print(f"Global Permutation p-value: {p_global:.4f} (Likely FP)")
    print(f"Stratified Permutation p-value: {p_strat:.4f} (Correctly Null)")

    if p_global < 0.05 and p_strat > 0.05:
        print("SUCCESS: Confounder control eliminated false positive.")
    else:
        print("WARNING: Confounder control did not behave as expected.")


if __name__ == "__main__":
    run_confounder_test()
