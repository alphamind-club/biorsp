"""
BioRSP Getting Started Example
==============================

This script demonstrates the core functionality of BioRSP using a synthetic dataset.
It covers:
1. Data generation and preprocessing
2. Defining a reference point (vantage point)
3. Finding spatially patterned genes (BioRSP analysis)
4. Interpretation and visualization
5. Stability analysis across embeddings
6. Baseline comparisons

Usage:
    python examples/getting_started.py
"""

import numpy as np
import pandas as pd
import scanpy as sc
import biorsp
from biorsp.baselines import run_baselines
from biorsp.interpretation import (
    generate_interpretation_report,
    identify_peak_sectors,
)
from biorsp.stability import consistency_across_embeddings
from examples.synthetic_data import create_synthetic_dataset

ALPHA = 0.05


def main():
    print("=== BioRSP Getting Started Example ===")

    print("\n[1] Generating synthetic dataset...")
    rng = np.random.default_rng(42)
    adata = create_synthetic_dataset(
        n_cells=2000, manifold="circle", n_genes=50, rng=rng
    )
    sc.pp.log1p(adata)
    print(f"    Dataset created: {adata.n_obs} cells, {adata.n_vars} genes")

    print("\n[2] Defining reference point (geometric median)...")
    vantage = biorsp.define_reference_point(adata, method="geometric_median")
    print(f"    Reference point defined at: {vantage}")

    print("\n[3] Running BioRSP analysis...")
    genes_to_scan = ["Gene_0", "Gene_5", "Gene_10"]
    print(f"    Scanning genes: {genes_to_scan}")

    results = biorsp.find_spatially_patterned_genes(
        adata,
        genes_to_test=genes_to_scan,
        reference_point=vantage,
        num_permutations=500,
        confounding_factors=["batch", "clusters"],
        neighbors_for_matching=30,
        check_spatial_distortion=False,
        random_seed=42,
        allow_uncalibrated_analysis=True,
    )
    print("    Analysis complete. Top results:")
    print(results[["ARIA", "p_CRA", "theta_dir"]].head())

    adata.uns["biorsp"]["params"] = {
        "vantage_point": vantage,
        "embedding_key": "X_umap",
    }

    print("\n[4] Generating interpretation report...")
    target_gene = "Gene_0"

    identify_peak_sectors(adata, target_gene, threshold_percentile=75)

    report = generate_interpretation_report(
        adata,
        [target_gene],
        obs_keys=["clusters", "batch"],
    )
    print(f"    Interpretation report generated for {target_gene}")

    print("\n[5] Running stability analysis across embeddings...")
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=2)

    stability_results = consistency_across_embeddings(
        adata,
        genes_to_test=[target_gene],
        coordinate_systems=["X_umap", "X_pca"],
        reference_method="geometric_median",
        analysis_settings={
            "allow_uncalibrated_analysis": True,
            "check_spatial_distortion": False,
            "confounding_factors": ["batch", "clusters"],
        },
    )
    print("    Stability results:")
    print(stability_results[["consistency_score", "passes_stability_test"]])

    print("\n[6] Running baseline methods for comparison...")
    baselines_df = run_baselines(adata, genes_to_scan)

    comparison = pd.concat(
        [results[["ARIA", "p_CRA"]], baselines_df],
        axis=1,
    )
    print("    Comparison with baselines:")
    print(comparison.head())

    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
