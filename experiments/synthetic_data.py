import numpy as np
import scanpy as sc
import anndata as ad


def generate_manifold(n_cells=1000, manifold_type="circle", noise=0.1, seed=42):
    """
    Generate coordinates for a specified manifold.

    Parameters
    ----------
    n_cells : int
        Number of cells to simulate.
    manifold_type : str
        'circle', 'y_branch', 'swiss_roll', 'blobs'
    noise : float
        Gaussian noise added to coordinates.
    seed : int
        Random seed.

    Returns
    -------
    coords : np.ndarray (n_cells, 2)
    t : np.ndarray (n_cells,) - underlying latent variable (e.g. angle or pseudotime)
    """
    np.random.seed(seed)

    if manifold_type == "circle":
        t = np.random.uniform(0, 2 * np.pi, n_cells)
        r = 5 + np.random.normal(0, noise, n_cells)
        x = r * np.cos(t)
        y = r * np.sin(t)
        coords = np.stack([x, y], axis=1)

    elif manifold_type == "y_branch":

        n_branch = n_cells // 3

        t1 = np.random.uniform(0, 5, n_branch)
        x1 = np.random.normal(0, noise, n_branch)
        y1 = t1 + np.random.normal(0, noise, n_branch)

        t2 = np.random.uniform(0, 5, n_branch)
        angle2 = np.radians(210)
        x2 = t2 * np.cos(angle2) + np.random.normal(0, noise, n_branch)
        y2 = t2 * np.sin(angle2) + np.random.normal(0, noise, n_branch)

        t3 = np.random.uniform(0, 5, n_cells - 2 * n_branch)
        angle3 = np.radians(330)
        x3 = t3 * np.cos(angle3) + np.random.normal(0, noise, n_cells - 2 * n_branch)
        y3 = t3 * np.sin(angle3) + np.random.normal(0, noise, n_cells - 2 * n_branch)

        coords = np.vstack(
            [
                np.stack([x1, y1], axis=1),
                np.stack([x2, y2], axis=1),
                np.stack([x3, y3], axis=1),
            ]
        )

        t = np.linalg.norm(coords, axis=1)

    elif manifold_type == "swiss_roll":
        t = np.random.uniform(1.5 * np.pi, 4.5 * np.pi, n_cells)
        x = t * np.cos(t)
        y = t * np.sin(t)
        coords = np.stack([x, y], axis=1) + np.random.normal(0, noise, (n_cells, 2))

    elif manifold_type == "blobs":
        n1 = n_cells // 2
        c1 = np.random.normal(loc=[-5, 0], scale=1, size=(n1, 2))
        c2 = np.random.normal(loc=[5, 0], scale=1, size=(n_cells - n1, 2))
        coords = np.vstack([c1, c2])
        t = np.zeros(n_cells)  # No meaningful continuous latent
        t[n1:] = 1

    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")

    return coords, t


def add_gene_expression(
    adata,
    coords,
    t,
    gene_type="directional",
    vantage_point=None,
    effect_size=2.0,
    dropout_rate=0.0,
    gene_name="Gene",
    clusters=None,
):
    """
    Add a gene to adata with specific expression pattern.
    """
    n_cells = adata.n_obs

    if gene_type == "directional":
        if vantage_point is None:
            vantage_point = np.mean(coords, axis=0)

        dx = coords[:, 0] - vantage_point[0]
        dy = coords[:, 1] - vantage_point[1]
        angles = np.arctan2(dy, dx)

        mu = 0
        k = effect_size

        expr = np.exp(k * np.cos(angles - mu))

    elif gene_type == "gradient":
        expr = coords[:, 0] * effect_size
        expr = expr - expr.min()  # Non-negative

    elif gene_type == "cluster_marker":
        if clusters is None:
            mask = coords[:, 0] > 0
        else:
            target_cluster = np.unique(clusters)[0]
            mask = clusters == target_cluster

        expr = np.zeros(n_cells)
        expr[mask] = effect_size * 5
        expr[~mask] = 0.5

    elif gene_type == "random":
        expr = np.random.exponential(1, n_cells)

    else:
        raise ValueError(f"Unknown gene type: {gene_type}")

    expr = expr / (expr.mean() + 1e-9) * 5

    counts = np.random.poisson(expr)

    if dropout_rate > 0:
        mask = np.random.random(n_cells) < dropout_rate
        counts[mask] = 0

    if gene_name in adata.var_names:
        adata.X[:, adata.var_names.get_loc(gene_name)] = counts
    else:
        pass

    return counts


def create_synthetic_dataset(n_cells=1000, manifold="circle", n_genes=100):
    """
    Create a full AnnData object with multiple gene types.
    """
    coords, t = generate_manifold(n_cells, manifold_type=manifold)

    X = np.zeros((n_cells, n_genes))
    var_names = [f"Gene_{i}" for i in range(n_genes)]
    obs_names = [f"Cell_{i}" for i in range(n_cells)]

    adata = ad.AnnData(X=X, dtype=np.float32)
    adata.obs_names = obs_names
    adata.var_names = var_names
    adata.obsm["X_umap"] = coords
    adata.obs["latent_t"] = t

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=3, random_state=42).fit(coords)
    clusters = kmeans.labels_.astype(str)
    adata.obs["clusters"] = clusters

    vantage = np.mean(coords, axis=0)
    for i in range(min(5, n_genes)):
        counts = add_gene_expression(
            adata,
            coords,
            t,
            "directional",
            vantage_point=vantage,
            effect_size=2.0 + i * 0.5,
            gene_name=f"Gene_{i}",
        )
        adata.X[:, i] = counts
        adata.var.loc[f"Gene_{i}", "ground_truth"] = "directional"

    for i in range(5, min(10, n_genes)):
        counts = add_gene_expression(
            adata,
            coords,
            t,
            "cluster_marker",
            effect_size=2.0,
            gene_name=f"Gene_{i}",
            clusters=clusters,
        )
        adata.X[:, i] = counts
        adata.var.loc[f"Gene_{i}", "ground_truth"] = "cluster_marker"

    for i in range(10, n_genes):
        counts = add_gene_expression(adata, coords, t, "random", gene_name=f"Gene_{i}")
        adata.X[:, i] = counts
        adata.var.loc[f"Gene_{i}", "ground_truth"] = "random"

    adata.obs["batch"] = np.random.choice(["Batch1", "Batch2"], n_cells)

    sc.pp.neighbors(adata, use_rep="X_umap")

    return adata
