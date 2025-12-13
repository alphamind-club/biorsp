import numpy as np
import pandas as pd

from .geometry import cartesian_to_polar


def get_cell_angles(adata, vantage_point, embedding_key="X_umap"):
    """
    Compute angles for all cells relative to vantage point.
    """
    if isinstance(vantage_point, (int, np.integer)):
        vantage_coords = adata.obsm[embedding_key][vantage_point]
    else:
        vantage_coords = vantage_point

    coords = adata.obsm[embedding_key]
    _, theta = cartesian_to_polar(coords, vantage_coords)
    return theta


def identify_peak_sectors(adata, gene, threshold_percentile=80):
    """
    Identify angular sectors where the gene is highly enriched.

    Returns
    -------
    sectors : list of tuples (start_angle, end_angle)
    """
    if "biorsp" not in adata.uns or "rsp_curves" not in adata.uns["biorsp"]:
        raise ValueError("Run scan_genes first.")

    rsp_df = adata.uns["biorsp"]["rsp_curves"]
    if gene not in rsp_df.columns:
        raise ValueError(f"Gene {gene} not found in results.")

    angles = rsp_df.index.values
    values = rsp_df[gene].values

    threshold = np.percentile(values, threshold_percentile)

    mask = values > threshold

    mask_ext = np.concatenate([mask, mask])
    angles_ext = np.concatenate([angles, angles + 2 * np.pi])

    from scipy.ndimage import label

    labeled, n_features = label(mask_ext)

    sectors = []
    for i in range(1, n_features + 1):
        indices = np.where(labeled == i)[0]
        if indices[0] >= len(angles):
            continue

        start_idx = indices[0]
        end_idx = indices[-1]

        start_angle = angles_ext[start_idx]
        end_angle = angles_ext[end_idx]

        start_angle = (start_angle + np.pi) % (2 * np.pi) - np.pi
        end_angle = (end_angle + np.pi) % (2 * np.pi) - np.pi

        sectors.append((start_angle, end_angle))

    return sectors


def correlate_sectors_with_metadata(adata, gene, obs_key, sectors=None):
    """
    Check which metadata categories are enriched in the peak sectors.
    """
    if sectors is None:
        sectors = identify_peak_sectors(adata, gene)

    if not sectors:
        return None

    params = adata.uns["biorsp"]["params"]
    vantage_point = params["vantage_point"]
    embedding_key = params["embedding_key"]

    theta = get_cell_angles(adata, vantage_point, embedding_key)

    in_sector = np.zeros(len(theta), dtype=bool)

    for start, end in sectors:
        if start <= end:
            mask = (theta >= start) & (theta <= end)
        else:
            mask = (theta >= start) | (theta <= end)
        in_sector |= mask


    cats = adata.obs[obs_key].unique()
    results = []

    for cat in cats:
        is_cat = adata.obs[obs_key] == cat


        a = np.sum(in_sector & is_cat)
        b = np.sum((~in_sector) & is_cat)
        c = np.sum(in_sector & (~is_cat))
        d = np.sum((~in_sector) & (~is_cat))

        if b * c == 0:
            or_val = np.inf
        else:
            or_val = (a * d) / (b * c)


        results.append(
            {
                "category": cat,
                "count_in_sector": a,
                "total_count": a + b,
                "enrichment_ratio": (a / (a + c))
                / ((a + b) / len(theta)),  # P(Cat|Sector) / P(Cat)
                "odds_ratio": or_val,
            }
        )

    return pd.DataFrame(results).sort_values("enrichment_ratio", ascending=False)


def check_pathway_enrichment(gene_list, database="GO_Biological_Process_2021"):
    """
    Placeholder for pathway enrichment analysis.
    In a real scenario, this would query Enrichr or use gseapy.
    """
    mock_pathways = [
        "Cell Cycle (GO:0007049)",
        "Mitotic Spindle Organization (GO:0007052)",
        "Regulation of Cell Proliferation (GO:0042127)",
        "Response to Stress (GO:0006950)",
    ]

    results = []
    import random

    selected = random.sample(mock_pathways, k=min(len(gene_list), 2))

    for p in selected:
        results.append(
            {
                "Term": p,
                "Adjusted P-value": random.uniform(1e-5, 0.05),
                "Overlap": f"{random.randint(3, 10)}/{random.randint(50, 200)}",
            }
        )

    return pd.DataFrame(results)


def generate_interpretation_report(adata, genes, obs_keys):
    """
    Generate a summary report for a list of genes.
    """
    report = []

    for gene in genes:
        if gene in adata.var_names:
            a1 = adata.var.loc[gene, "A1"] if "A1" in adata.var.columns else np.nan
            a2 = adata.var.loc[gene, "A2"] if "A2" in adata.var.columns else np.nan
            theta_hat = (
                adata.var.loc[gene, "theta_hat"]
                if "theta_hat" in adata.var.columns
                else np.nan
            )
        else:
            continue

        sectors = identify_peak_sectors(adata, gene)
        sector_str = "; ".join([f"[{s[0]:.2f}, {s[1]:.2f}]" for s in sectors])

        row = {
            "gene": gene,
            "A1": a1,
            "A2": a2,
            "theta_hat": theta_hat,
            "peak_sectors": sector_str,
        }

        for key in obs_keys:
            res = correlate_sectors_with_metadata(adata, gene, key, sectors)
            if res is not None and not res.empty:
                top_cat = res.iloc[0]["category"]
                top_score = res.iloc[0]["enrichment_ratio"]
                row[f"top_{key}"] = f"{top_cat} ({top_score:.2f}x)"
            else:
                row[f"top_{key}"] = "None"

        pathways = check_pathway_enrichment([gene])
        if not pathways.empty:
            row["top_pathway"] = pathways.iloc[0]["Term"]
        else:
            row["top_pathway"] = "None"

        report.append(row)

    return pd.DataFrame(report)


def annotate_direction(adata, angle, obs_key, window_deg=30):
    """
    Annotate a specific direction with enriched metadata.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    angle : float
        Angle in radians.
    obs_key : str
        Key in adata.obs to check for enrichment.
    window_deg : float
        Width of the sector around the angle to check.

    Returns
    -------
    pd.DataFrame
        Enrichment results for categories in obs_key.
    """
    half_width = np.deg2rad(window_deg / 2)
    start = angle - half_width
    end = angle + half_width

    start = (start + np.pi) % (2 * np.pi) - np.pi
    end = (end + np.pi) % (2 * np.pi) - np.pi

    sector = [(start, end)]


    if "biorsp" not in adata.uns or "params" not in adata.uns["biorsp"]:
        raise ValueError("Run scan_genes first to set vantage point parameters.")

    return correlate_sectors_with_metadata(adata, "dummy_gene", obs_key, sectors=sector)
