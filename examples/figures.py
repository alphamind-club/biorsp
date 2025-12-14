"""Aggregate BioRSP discovery CSVs and generate summary figures.

Usage:
    python3 examples/figures.py

Outputs saved under `examples/outputs/figures/`.
"""

import glob
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIG_DIR = "examples/outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def format_condition_label(cond: str) -> str:
    """Format condition names for display.
    
    Maps common condition patterns to abbreviations:
    - acute_kidney_failure -> AKI
    - chronic -> CKD
    - normal -> Normal
    """
    label_map = {
        "acute_kidney_failure": "AKI",
        "chronic_kidney_disease": "CKD",
        "normal": "Normal",
    }
    return label_map.get(cond, cond)


def format_metric_label(metric: str) -> str:
    """Format metric names for display.
    
    Replaces underscores with spaces.
    """
    return metric.replace('_', ' ')


def load_discoveries(outputs_dir: str = "examples/outputs") -> Dict[str, pd.DataFrame]:
    """Load per-condition discovery CSVs into a dict mapping condition -> DataFrame.

    DataFrames are indexed by `ensg_id` when available and should contain a `gene` column
    with the human-readable feature name. This preserves ENSG identifiers internally but
    surfaces feature names in plots when possible.
    """
    pattern = os.path.join(outputs_dir, "*", "*_discoveries.csv")
    files = glob.glob(pattern)
    data = {}
    for fp in files:
        condition = os.path.basename(os.path.dirname(fp))
        try:
            df = pd.read_csv(fp)
            if "ensg_id" in df.columns:
                df = df.set_index("ensg_id")
            if "gene" not in df.columns:
                df["gene"] = df.index.to_series().astype(str)
            data[condition] = df
        except Exception as e:
            print(f"Failed to read {fp}: {e}")
    return data


def make_metric_matrix(
    data: Dict[str, pd.DataFrame], metric: str = "ARIA"
) -> pd.DataFrame:
    """Return a DataFrame with rows=feature names (when available), columns=conditions, values=metric.

    If multiple ENSG IDs map to the same feature name, rows are aggregated by taking the maximum
    metric value across those ENSGs to preserve top signals.
    """
    frames = {}
    gene_map = {}
    for cond, df in data.items():
        if metric in df.columns:
            series = df[metric].rename(cond)
            frames[cond] = series
            # collect mapping from ensg -> feature name
            if "gene" in df.columns:
                gene_map.update(df["gene"].to_dict())
    if not frames:
        raise ValueError("No metric columns found in any discovery files.")
    mat = pd.DataFrame(frames)

    mapped_index = mat.index.map(lambda ensg: gene_map.get(ensg, ensg))
    mat.index = mapped_index

    if mat.index.duplicated().any():
        mat = mat.groupby(mat.index).max()

    mat.columns = [format_condition_label(col) for col in mat.columns]

    return mat


def heatmap(mat: pd.DataFrame, outpath: str, cmap: str = "vlag"):
    plt.figure(figsize=(14, max(10, 0.5 * len(mat))))
    sns.heatmap(mat, cmap=cmap, center=0, linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.xlabel("Conditions", fontsize=24)
    plt.ylabel("Features", fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def clustermap(mat: pd.DataFrame, outpath: str, cmap: str = "vlag"):
    g = sns.clustermap(mat.fillna(0), cmap=cmap, center=0, figsize=(16, 16))
    g.ax_heatmap.set_xlabel("Conditions", fontsize=24)
    g.ax_heatmap.set_ylabel("Features", fontsize=24)
    g.ax_heatmap.tick_params(axis='x', labelsize=20)
    g.ax_heatmap.tick_params(axis='y', labelsize=20)
    plt.savefig(outpath)
    plt.close()


def bar_top_genes(
    data: Dict[str, pd.DataFrame],
    metric: str = "CRA W1",
    top_n: int | None = None,
    outpath: str = None,
):
    rows = []
    for cond, df in data.items():
        if metric in df.columns:
            top = df.sort_values(metric, ascending=False)
            if top_n is not None:
                top = top.head(top_n)
            for idx, row in top.iterrows():
                rows.append(
                    {
                        "condition": format_condition_label(cond),
                        "ensg": idx,
                        "metric": row[metric],
                        "gene": row.get("gene", idx),
                    }
                )
    df_all = pd.DataFrame(rows)
    if df_all.empty:
        print("No top genes available for bar plot.")
        return
    plt.figure(figsize=(14, 8))
    sns.barplot(x="condition", y="metric", hue="gene", data=df_all, order=["Normal", "AKI", "CKD"])
    plt.xlabel("Condition", fontsize=24)
    plt.ylabel(f"{format_metric_label(metric)} Value", fontsize=24)
    plt.xticks(rotation=30, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
    plt.close()


def scatter_metrics(
    data: Dict[str, pd.DataFrame],
    x: str = "CRA W1",
    y: str = "Pattern Strength CV",
    top_n: int | None = None,
    outpath: str = None,
):
    rows = []
    for cond, df in data.items():
        if x in df.columns and y in df.columns:
            tmp = df[[x, y]].copy()
            tmp = tmp.reset_index().rename(columns={"index": "ensg"})
            if top_n is not None:
                top_idx = df.sort_values(x, ascending=False).head(top_n).index
                tmp = tmp[tmp["ensg"].isin(top_idx)]
            tmp["gene"] = tmp["ensg"].map(
                lambda ens: (
                    df.loc[ens, "gene"]
                    if "gene" in df.columns and ens in df.index
                    else ens
                )
            )
            tmp["condition"] = format_condition_label(cond)
            rows.append(tmp)
    if not rows:
        print("No data available for scatter plot.")
        return
    df_all = pd.concat(rows, ignore_index=True)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_all, x=x, y=y, hue="condition", alpha=0.7, hue_order=["Normal", "AKI", "CKD"])
    plt.xlabel(format_metric_label(x), fontsize=24)
    plt.ylabel(format_metric_label(y), fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
    plt.close()


def venn_top_genes(data: Dict[str, pd.DataFrame], top_n: int | None = None, outpath: str = None):
    conds = list(data.keys())
    conds = [format_condition_label(c) for c in conds]
    sets = []
    for df in data.values():
        if "ARIA" not in df.columns:
            continue
        top_idx = df.sort_values("ARIA", ascending=False)
        if top_n is not None:
            top_idx = top_idx.head(top_n)
        top_idx = top_idx.index
        names = [
            (
                df.loc[ens, "gene"]
                if ("gene" in df.columns and ens in df.index)
                else str(ens)
            )
            for ens in top_idx
        ]
        sets.append(set(names))

    if len(sets) == 0:
        print("Not enough sets for venn/upset diagram.")
        return

    if len(sets) <= 3:
        try:
            from matplotlib_venn import venn2, venn3
        except Exception:
            print("matplotlib-venn not available; skipping venn diagrams.")
            return
        if len(sets) == 2:
            plt.figure(figsize=(6, 6))
            venn2(sets, set_labels=conds[:2])
            if outpath:
                plt.savefig(outpath)
            plt.close()
        else:
            plt.figure(figsize=(6, 6))
            venn3(sets[:3], set_labels=conds[:3])
            if outpath:
                plt.savefig(outpath)
            plt.close()
        return

    try:
        from upsetplot import UpSet, from_indicators
    except Exception:
        print("upsetplot not available; attempting venn on first 3 sets as fallback.")
        try:
            from matplotlib_venn import venn3

            plt.figure(figsize=(6, 6))
            venn3(sets[:3], set_labels=conds[:3])
            if outpath:
                plt.savefig(outpath)
            plt.close()
        except Exception:
            print("Venn fallback failed; skipping set-intersection diagram.")
        return

    union_genes = sorted(set().union(*sets))
    ind = pd.DataFrame(index=union_genes)
    for cond, s in zip(conds, sets):
        ind[cond] = [1 if g in s else 0 for g in ind.index]

    ups = from_indicators(ind.astype(bool))
    fig = plt.figure(figsize=(10, 6))
    UpSet(ups).plot(fig=fig)
    if outpath:
        base, ext = os.path.splitext(outpath)
        upset_out = f"{base.replace('venn', 'upset')}{ext if ext else '.png'}"
        plt.savefig(upset_out)
    plt.close()


def boxplot_metric(
    data: Dict[str, pd.DataFrame], metric: str = "CRA W1", top_n: int | None = None, outpath: str = None
):
    rows = []
    for cond, df in data.items():
        if metric in df.columns:
            if top_n is not None:
                top_idx = df.sort_values(metric, ascending=False).head(top_n).index
                tmp = df.loc[top_idx, [metric]].copy()
            else:
                tmp = df[[metric]].copy()
            tmp = tmp.reset_index().rename(columns={"index": "ensg"})
            tmp["condition"] = format_condition_label(cond)
            rows.append(tmp)
    if not rows:
        print("No data available for boxplot.")
        return
    df_all = pd.concat(rows, ignore_index=True)
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=df_all, x="condition", y=metric, order=["Normal", "AKI", "CKD"])
    plt.xlabel("Condition", fontsize=24)
    plt.ylabel(format_metric_label(metric), fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate summary figures from BioRSP discovery CSVs"
    )
    parser.add_argument(
        "--outputs-dir",
        default="examples/outputs",
        help="Path to examples outputs directory",
    )
    parser.add_argument("--out-dir", default=FIG_DIR, help="Path to save figures")
    parser.add_argument(
        "--metric",
        default="ARIA",
        help="Metric column to use for heatmap/bar/venn (default: ARIA)",
    )
    parser.add_argument(
        "--heatmap-top",
        type=int,
        default=200,
        help="Number of top genes to keep for heatmap (by max across conditions)",
    )
    parser.add_argument(
        "--top-n",
        type=str,
        default='50',
        help="Top N genes per condition for bar/venn plots, or 'all' to use all genes",
    )
    parser.add_argument(
        "--scatter-y", default="pattern_strength_cv", help="Y metric for scatter plot"
    )
    args = parser.parse_args()

    if args.top_n == 'all':
        top_n = None
    else:
        top_n = int(args.top_n)

    data = load_discoveries(outputs_dir=args.outputs_dir)
    if not data:
        print(
            "No discovery CSVs found in examples/outputs; run `examples/example_full.py` first."
        )
        return

    FIG_DIR_ACTUAL = args.out_dir
    os.makedirs(FIG_DIR_ACTUAL, exist_ok=True)

    try:
        mat = make_metric_matrix(data, metric=args.metric)
    except ValueError as e:
        print(e)
        return

    top_genes = (
        mat.max(axis=1).sort_values(ascending=False).head(args.heatmap_top).index
    )
    mat_top = mat.loc[top_genes]

    if top_n is not None:
        heatmap(mat_top, os.path.join(FIG_DIR_ACTUAL, f"{args.metric}_heatmap.png"))
        clustermap(mat_top, os.path.join(FIG_DIR_ACTUAL, f"{args.metric}_clustermap.png"))

    bar_top_genes(
        data,
        metric=args.metric,
        top_n=top_n,
        outpath=os.path.join(FIG_DIR_ACTUAL, "top_genes_bar.png"),
    )
    scatter_metrics(
        data,
        x=args.metric,
        y=args.scatter_y,
        top_n=top_n,
        outpath=os.path.join(FIG_DIR_ACTUAL, "scatter_metrics.png"),
    )
    venn_top_genes(
        data, top_n=top_n, outpath=os.path.join(FIG_DIR_ACTUAL, "venn_top50.png")
    )
    boxplot_metric(
        data,
        metric=args.metric,
        top_n=top_n,
        outpath=os.path.join(FIG_DIR_ACTUAL, f"{args.metric}_boxplot.png"),
    )

    print(f"Saved figures to {FIG_DIR_ACTUAL}")


if __name__ == "__main__":
    main()
