import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from examples.synthetic_data import (
    add_gene_expression,
    create_synthetic_dataset,
    generate_manifold,
)


def test_generate_manifold_rng_consistency():
    rng = np.random.default_rng(0)
    coords1, t1 = generate_manifold(100, manifold_type="circle", seed=0, rng=rng)
    rng2 = np.random.default_rng(0)
    coords2, t2 = generate_manifold(100, manifold_type="circle", seed=0, rng=rng2)
    assert np.allclose(coords1, coords2)
    assert np.allclose(t1, t2)


def test_create_dataset_and_gene_expression():
    rng = np.random.default_rng(42)
    n_cells = 100
    n_genes = 10

    adata = create_synthetic_dataset(
        n_cells=n_cells,
        manifold="circle",
        n_genes=n_genes,
        rng=rng,
    )
    assert adata.n_obs == n_cells
    assert adata.n_vars == n_genes

    counts = add_gene_expression(
        adata,
        adata.obsm["X_umap"],
        adata.obs["latent_t"],
        gene_type="random",
        gene_name="Gene_0",
        rng=rng,
    )
    assert len(counts) == adata.n_obs
