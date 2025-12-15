# BioRSP Best Practices Guide

## Interpreting `consistency_across_embeddings`

BioRSP provides a `consistency_across_embeddings` metric (or similar stability checks) to ensure that discovered spatial patterns are robust to the choice of dimensionality reduction (e.g., UMAP vs. PCA vs. t-SNE).

- **High Consistency (> 0.8):** The gene's spatial pattern is stable and likely biological. The radial gradient is preserved across different manifold representations.
- **Low Consistency (< 0.5):** The pattern may be an artifact of the specific embedding (e.g., UMAP distortion).
  - **Action:** Verify the result using `check_spatial_distortion=True`.
  - **Action:** Try running BioRSP on the physical spatial coordinates (if available) or a linear embedding like PCA.

## Handling Stability Test Failures

BioRSP performs a "Vantage Point Stability Test" (`reference_point_stability_tests`) to ensure the center of the radial pattern is well-defined.

### What if a gene fails the stability test?

If you see warnings like `Vantage point instability detected`, it means the geometric center of the dataset shifts significantly when the data is subsampled.

**Recommended Actions:**

1.  **Increase Cell Count:** Small datasets (< 200 cells) often have unstable centers.
2.  **Use a Fixed Reference:** Instead of `method="geometric_median"`, manually specify a robust reference point using `define_reference_point(..., method="coordinates")` based on biological knowledge (e.g., a known landmark).
3.  **Relax Thresholds:** For exploratory analysis, you can increase `stability_threshold_fraction` (default 0.05) to 0.10, but be cautious about over-interpreting results.

## Parallelization

BioRSP supports parallel processing for gene scanning.

- Use `n_jobs=-1` to use all available cores.
- Use `n_jobs=1` (default) for debugging or small datasets.

## Permutation Methods

- **`stratified`:** Best for discrete confounders (e.g., Batch, Sample). Fast and robust.
- **`knn`:** Best for continuous confounders (e.g., Library Size, Cell Cycle Score). Computationally more expensive but more flexible.
