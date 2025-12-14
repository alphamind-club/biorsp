### BioRSP

[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3765612.3767793-blue)](https://doi.org/10.1145/3765612.3767793)
[![Version](https://img.shields.io/badge/version-3.0.0--dev-orange)]()

BioRSP identifies genes with directional expression gradients in 2D single-cell embeddings, controlling for technical confounders.

#### Installation

```bash
pip install git+https://github.com/alphamind-club/biorsp.git
```

**Requirements:** Python 3.8+, scanpy, numpy, scipy, scikit-learn, pandas, tqdm

#### Quick Start

```python
import scanpy as sc
import biorsp

# Load and subset data
adata = sc.read_h5ad("data.h5ad")
adata = adata[adata.obs['condition'] == 'target']

# Define vantage point
vantage = biorsp.define_reference_point(adata, coordinate_system="X_umap", method="geometric_median")

# Run analysis
results = biorsp.find_spatially_patterned_genes(
    adata,
    genes_to_test=list(adata.var_names),
    reference_point=vantage,
    coordinate_system="X_umap",
    confounding_factors=["sample_id", "batch"],
    num_permutations=500,
    min_cells_required=200
)

# Filter discoveries
discoveries = results[results['discovery']].sort_values('ARIA', ascending=False)
```

#### Core Method

BioRSP measures the **Anisotropy Radial Index Analysis (ARIA)** score as the circular Wasserstein-1 distance between observed and expected angular distributions under covariate-conditional nulls.

**Key Features:**

- Gene-independent permutation indices for valid multiplicity control and 10–100× speedup
- Fail-closed guardrails: minimum cell counts, vantage stability, embedding distortion checks
- Multi-embedding stability gate for adoption-grade discoveries
- Standardized Z-scores for cross-dataset comparison

#### Examples

Runnable scripts in `examples/`:

- `examples.py`: Basic workflow with synthetic data
- `figures.py`: Generate summary figures
- `marker_genes_tal.py`: Real kidney data analysis
- Performance/validation suites: `run_*.py` scripts

Run with: `python -m examples.examples`

#### Documentation

- **Mathematical Specification:** [SPEC.md](SPEC.md)
- **API Reference:** Inline docstrings
- **Issues:** [GitHub Issues](https://github.com/alphamind-club/biorsp/issues)

#### License

MIT License. See [LICENSE](LICENSE).
