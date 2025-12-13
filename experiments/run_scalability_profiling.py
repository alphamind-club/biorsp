import time
import biorsp
from experiments.synthetic_data import create_synthetic_dataset


def run_scalability_profiling():
    print("Running Scalability Profiling...")

    sizes = [1000, 5000, 10000]  # Keep small for demo, scale up for real paper
    times = []

    for n in sizes:
        print(f"  Profiling N={n} cells...")
        adata = create_synthetic_dataset(n_cells=n, n_genes=10)
        vantage = biorsp.set_vantage(adata, mode="geometric_median")

        start = time.time()
        biorsp.scan_genes(adata, ["Gene_0"], vantage, n_perm=50)
        end = time.time()

        times.append(end - start)
        print(f"    Time: {end - start:.2f}s")

    print("\nScalability Results:")
    for n, t in zip(sizes, times):
        print(f"N={n}: {t:.2f}s")

    ratios = [times[i] / times[i - 1] for i in range(1, len(times))]
    size_ratios = [sizes[i] / sizes[i - 1] for i in range(1, len(sizes))]

    print("\nScaling Factors:")
    for r, sr in zip(ratios, size_ratios):
        print(f"Time Ratio: {r:.2f} (Size Ratio: {sr:.2f})")


if __name__ == "__main__":
    run_scalability_profiling()
