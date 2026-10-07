#!/usr/bin/env python3
"""Plot ancestry distributions for arbitrary sample groups and 2 or 3 ancestry components."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .common import read_population_map, validate_population_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="ancestry_summary.tsv")
    parser.add_argument("--population-map", required=True)
    parser.add_argument("--state-order", nargs="+", required=True, metavar="POP")
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--out-prefix", default="ancestry_distributions")
    args = parser.parse_args()
    try:
        validate_population_names(args.state_order); mapping = read_population_map(args.population_map)
    except ValueError as exc: parser.error(str(exc))
    data = pd.read_csv(args.summary, sep="\t", dtype={"sample": str})
    required = {"sample", *args.state_order}
    if not required.issubset(data.columns): parser.error(f"Summary must contain {sorted(required)}")
    if data.empty or data["sample"].isna().any() or data["sample"].duplicated().any(): parser.error("Summary must contain unique nonempty sample IDs")
    values = data[args.state_order].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-5):
        parser.error("Ancestry proportions must be finite, within 0-1, and sum to 1")
    data["population"] = data["sample"].map(mapping)
    unknown = data.loc[data["population"].isna(), "sample"].tolist()
    if unknown: parser.error(f"Samples missing from population map: {unknown}")
    groups = args.groups or list(dict.fromkeys(mapping.values())); data = data[data["population"].isin(groups)].copy()
    counts = data["population"].value_counts(); missing_groups = [g for g in groups if counts.get(g, 0) == 0]
    if missing_groups: parser.error(f"No summarized samples found for groups: {missing_groups}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, len(args.state_order), figsize=(max(7, 3.3 * len(args.state_order)), 4.3), sharey=True, squeeze=False)
    axes = axes[0]; rng = np.random.default_rng(42); cmap = plt.get_cmap("tab10"); colors = {g: cmap(i % 10) for i, g in enumerate(groups)}
    for ax, ancestry in zip(axes, args.state_order):
        for x, group in enumerate(groups):
            vals = data.loc[data["population"] == group, ancestry].to_numpy()
            if len(vals) > 1 and np.ptp(vals) > 1e-12:
                violin = ax.violinplot(vals, positions=[x], widths=0.7, showextrema=False, points=150)
                for body in violin["bodies"]: body.set_facecolor(colors[group]); body.set_edgecolor(colors[group]); body.set_alpha(0.3)
            ax.scatter(x + rng.uniform(-0.12, 0.12, len(vals)), vals, s=12, alpha=0.55, color=colors[group], edgecolors="none", rasterized=True, zorder=3)
            ax.plot([x - 0.18, x + 0.18], [np.median(vals)] * 2, color="black", lw=1.8, zorder=4)
        ax.set_title(f"{ancestry} ancestry"); ax.set_xticks(range(len(groups)), [f"{g}\n(n={counts[g]})" for g in groups]); ax.set_xlabel("Sample group")
        ax.set_xlim(-0.6, len(groups) - 0.4); ax.set_ylim(-0.025, 1.025); ax.set_yticks(np.linspace(0, 1, 6)); ax.grid(axis="y", alpha=0.2); ax.set_axisbelow(True)
    axes[0].set_ylabel("Inferred ancestry proportion"); fig.tight_layout()
    prefix = Path(args.out_prefix); prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        output = Path(str(prefix) + "." + ext); fig.savefig(output, dpi=300, bbox_inches="tight"); print(f"Saved {output}")
    plt.close(fig)


if __name__ == "__main__": main()
