import time
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import biorsp
from synthetic_data import create_synthetic_dataset


def linear_func(x, a, b):
    return a * x + b


def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c


def run_scalability_profiling():
    fast_val = os.environ.get("FAST_PROFILE", "0")
    fast_mode = fast_val in ("1", "true", "True")

    if fast_mode:
        sizes = [100, 500, 1000]
    else:
        sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    times = []
    print(f"Running scalability profiling for N={sizes}...")

    for n in sizes:
        print(f"  Profiling N={n}...")
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
                allow_exploratory_mode=True,
                permutation_method="knn",
                confounding_factors=[],
                check_spatial_distortion=False,
                n_jobs=1,
            )
            end = time.time()
            duration = end - start
            times.append(duration)
            print(f"    Time: {duration:.4f}s")
        except BioRSPValidationError as exc:
            print(f"    Run for n={n} failed validation: {exc}")
            times.append(None)
            continue
        except Exception as e:
            print(f"    Run for n={n} failed with error: {e}")
            times.append(None)
            continue

    valid_sizes = []
    valid_times = []
    for s, t in zip(sizes, times):
        if t is not None:
            valid_sizes.append(s)
            valid_times.append(t)

    if len(valid_sizes) < 3:
        print("Not enough data points to fit curve.")
        return

    valid_sizes = np.array(valid_sizes)
    valid_times = np.array(valid_times)

    popt_lin, _ = curve_fit(linear_func, valid_sizes, valid_times)
    popt_quad, _ = curve_fit(quadratic_func, valid_sizes, valid_times)

    residuals_lin = valid_times - linear_func(valid_sizes, *popt_lin)
    ss_res_lin = np.sum(residuals_lin**2)
    ss_tot = np.sum((valid_times - np.mean(valid_times)) ** 2)
    r2_lin = 1 - (ss_res_lin / ss_tot)

    residuals_quad = valid_times - quadratic_func(valid_sizes, *popt_quad)
    ss_res_quad = np.sum(residuals_quad**2)
    r2_quad = 1 - (ss_res_quad / ss_tot)

    print("\nComplexity Analysis:")
    print(f"  Linear Fit R^2: {r2_lin:.4f}")
    print(f"  Quadratic Fit R^2: {r2_quad:.4f}")

    if r2_lin > r2_quad:
        print("  Conclusion: Empirical complexity is approximately Linear O(N).")
    else:
        print("  Conclusion: Empirical complexity may be super-linear.")

    plt.figure(figsize=(10, 6))
    plt.scatter(valid_sizes, valid_times, label="Measured Times", color="blue")

    x_range = np.linspace(min(valid_sizes), max(valid_sizes), 100)
    plt.plot(
        x_range,
        linear_func(x_range, *popt_lin),
        "r--",
        label=f"Linear Fit ($R^2={r2_lin:.3f}$)",
    )
    plt.plot(
        x_range,
        quadratic_func(x_range, *popt_quad),
        "g:",
        label=f"Quadratic Fit ($R^2={r2_quad:.3f}$)",
    )

    plt.xlabel("Number of Cells (N)")
    plt.ylabel("Execution Time (s)")
    plt.title("BioRSP Scalability: Time vs N")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join("examples", "scalability_plot.png")
    plt.savefig(out_path)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    run_scalability_profiling()


if __name__ == "__main__":
    run_scalability_profiling()
