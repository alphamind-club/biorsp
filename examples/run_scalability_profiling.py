import time

import biorsp
from examples.synthetic_data import create_synthetic_dataset


def run_scalability_profiling():
    sizes = [1000, 5000, 10000]
    times = []

    for n in sizes:
        adata = create_synthetic_dataset(n_cells=n, n_genes=10)
        vantage = biorsp.define_reference_point(adata, method="geometric_median")

        start = time.time()
        biorsp.find_spatially_patterned_genes(
            adata,
            genes_to_test=["Gene_0"],
            reference_point=vantage,
            coordinate_system="X_umap",
            num_permutations=50,
            allow_uncalibrated_analysis=True,
            permutation_method="knn",
            check_spatial_distortion=False,
        )
        end = time.time()

        times.append(end - start)

    ratios = [times[i] / times[i - 1] for i in range(1, len(times))]
    size_ratios = [sizes[i] / sizes[i - 1] for i in range(1, len(sizes))]

    print("Times:", times)
    print("Ratios:", ratios)
    print("Size ratios:", size_ratios)


if __name__ == "__main__":
    run_scalability_profiling()
