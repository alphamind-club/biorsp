import time

import biorsp
from examples.synthetic_data import create_synthetic_dataset


import os


def run_scalability_profiling()
    fast_val = os.environ.get("FAST_PROFILE", "1")  # default to quick runs
    fast_mode = fast_val in ("1", "true", "True")
    if fast_mode:
        sizes = [100, 500, 1000]
    else:
        sizes = [1000, 5000, 10000]

    times = []

    for n in sizes:
        adata = create_synthetic_dataset(n_cells=n, n_genes=10)
        vantage = biorsp.define_reference_point(adata, method="geometric_median")

        from biorsp.validation import BioRSPValidationError

        start = time.time()
        try:
            biorsp.find_spatially_patterned_genes(
                adata,
                genes_to_test=["Gene_0"],
                reference_point=vantage,
                coordinate_system="X_umap",
                num_permutations=50,
                allow_uncalibrated_analysis=True,
                allow_exploratory_mode=fast_mode,
                permutation_method="knn",
                check_spatial_distortion=False,
            )
            end = time.time()
            times.append(end - start)
        except BioRSPValidationError as exc:
            print(f"Run for n={n} failed validation: {exc}")
            times.append(None)
            continue

    # Compute ratios only for successful (non-None) runs and corresponding sizes
    successful = [(s, t) for s, t in zip(sizes, times) if t is not None]
    if len(successful) < 2:
        print("Not enough successful runs to compute scaling ratios.")
        print("Times:", times)
        return

    success_sizes, success_times = zip(*successful)

    ratios = [success_times[i] / success_times[i - 1] for i in range(1, len(success_times))]
    size_ratios = [success_sizes[i] / success_sizes[i - 1] for i in range(1, len(success_sizes))]

    print("Times:", times)
    print("Ratios:", ratios)
    print("Size ratios:", size_ratios)


if __name__ == "__main__":
    run_scalability_profiling()
