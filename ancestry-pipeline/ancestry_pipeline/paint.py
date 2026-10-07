#!/usr/bin/env python3
"""Paint local ancestry for any configured ancestry population in a 2- or 3-population run."""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from .common import diploid_states, read_chromosome_map, read_population_map, validate_population_names


def read_posterior(path, state_order, chromosome_map=None):
    states = diploid_states(len(state_order))
    copies = np.array([[int(x) for x in state.split(",")] for state in states], dtype=float) / 2
    data = pd.read_csv(path, sep=r"\s+", dtype={"chrom": str})
    required = {"chrom", "position", *states}
    if not required.issubset(data.columns) or data.empty:
        raise ValueError(f"{path}: missing posterior columns or empty data")
    probabilities = data[states].to_numpy(dtype=float)
    totals = probabilities.sum(axis=1)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(totals, 1, atol=0.001, rtol=0):
        raise ValueError(f"{path}: invalid posterior probabilities")
    fractions = probabilities / totals[:, None] @ copies
    for index, population in enumerate(state_order):
        data[population] = fractions[:, index]
    data["position"] = pd.to_numeric(data["position"], errors="raise")
    for segment, group in data.groupby("chrom", sort=False):
        if (np.diff(group["position"].to_numpy()) <= 0).any():
            raise ValueError(f"{path}: unsorted or duplicate positions in {segment}")
    chromosome_map = chromosome_map or {}
    def label(segment):
        base = str(segment).split("_seg")[0]
        return chromosome_map.get(base, base)
    data["chromosome_label"] = data["chrom"].map(label)
    return data


def paint_sample(data, sample, focus_population, global_value, state_order, chromosome_order=None):
    present = list(dict.fromkeys(data["chromosome_label"]))
    if chromosome_order:
        chromosomes = [x for x in chromosome_order if x in present] + [x for x in present if x not in chromosome_order]
    else:
        chromosomes = present
    fig, axes = plt.subplots(len(chromosomes), 1, figsize=(11, max(4.5, len(chromosomes) * 1.05 + 1.5)), sharex=True, squeeze=False)
    max_mb = data["position"].max() / 1e6
    cmap = plt.get_cmap("tab10")
    colors = {pop: cmap(i) for i, pop in enumerate(state_order)}
    for ax, chrom in zip(axes[:, 0], chromosomes):
        for _, segment in data[data["chromosome_label"] == chrom].groupby("chrom", sort=False):
            x = segment["position"].to_numpy() / 1e6
            values = [segment[pop].to_numpy() for pop in state_order]
            if len(x) >= 2:
                ax.stackplot(x, *values, colors=[colors[p] for p in state_order], linewidth=0, rasterized=True)
            else:
                ends = np.r_[0, np.cumsum([v[0] for v in values])]
                for i, pop in enumerate(state_order):
                    ax.vlines(x[0], ends[i], ends[i + 1], color=colors[pop], lw=1.5)
        ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", ".5", "1"])
        ax.text(-0.075, 0.5, chrom, transform=ax.transAxes, ha="right", va="center", fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=9)
    axes[-1, 0].set_xlim(0, max_mb * 1.015); axes[-1, 0].set_xlabel("Position (Mb)")
    fig.suptitle(f"{sample}  |  Global {focus_population} ancestry: {global_value:.1%}", y=0.975, fontsize=14)
    fig.legend(handles=[Patch(facecolor=colors[p], label=p) for p in state_order], loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=len(state_order), frameon=False)
    fig.text(0.02, 0.5, "Ancestry proportion", rotation=90, va="center", fontsize=11)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.84, bottom=0.11, hspace=0.45)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", required=True, help="Ancestry population used for high/low selection")
    parser.add_argument("--group", choices=["high", "low"], required=True)
    parser.add_argument("--summary", default="ancestry_summary.tsv")
    parser.add_argument("--posterior-dir", default="ancestry_hmm_results/posteriors")
    parser.add_argument("--population-map", required=True, help="SAMPLE_ID POPULATION table for selecting samples from the requested group")
    parser.add_argument("--sample-group", help="Sample population/group to paint; default is the same name as --population")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--state-order", nargs="+", required=True, metavar="POP")
    parser.add_argument("--chromosome-map", help="Optional two-column CHROM LABEL table for display labels/order")
    parser.add_argument("--out")
    args = parser.parse_args()
    try:
        validate_population_names(args.state_order)
        mapping = read_population_map(args.population_map)
        chrom_map, chrom_order = read_chromosome_map(args.chromosome_map)
    except ValueError as exc:
        parser.error(str(exc))
    if args.population not in args.state_order:
        parser.error("--population must be one of --state-order")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be within 0-1")
    sample_group = args.sample_group or args.population
    summary = pd.read_csv(args.summary, sep="\t", dtype={"sample": str})
    if not {"sample", args.population}.issubset(summary.columns):
        parser.error(f"Summary must contain sample and {args.population} columns")
    summary[args.population] = pd.to_numeric(summary[args.population], errors="raise")
    summary["sample_group"] = summary["sample"].map(mapping)
    missing_map = summary.loc[summary["sample_group"].isna(), "sample"].tolist()
    if missing_map:
        parser.error(f"Samples missing from population map: {missing_map}")
    ancestry_mask = summary[args.population] > args.threshold if args.group == "high" else summary[args.population] < args.threshold
    selected = summary[summary["sample_group"].eq(sample_group) & ancestry_mask].copy()
    if selected.empty:
        parser.error(f"No samples in group {sample_group!r} satisfy {args.group} {args.population} ancestry threshold")
    if "state_order" in selected.columns and not selected["state_order"].eq(",".join(args.state_order)).all():
        parser.error("Summary state_order differs from --state-order")
    folder = Path(args.posterior_dir)
    paths = [folder / f"{sample}.posterior" for sample in selected["sample"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error("Missing posterior files:\n" + "\n".join(missing))
    output = Path(args.out or f"{args.group}_{sample_group}_{args.population}_chromosome_paintings.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    temporary = output.with_name(output.name + ".tmp")
    try:
        with PdfPages(temporary) as pdf:
            for (_, row), path in zip(selected.iterrows(), paths):
                data = read_posterior(path, args.state_order, chrom_map)
                fig = paint_sample(data, row["sample"], args.population, row[args.population], args.state_order, chrom_order)
                pdf.savefig(fig); plt.close(fig); print(f"Painted {row['sample']}", flush=True)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Saved {len(selected)} sample pages to {output}")


if __name__ == "__main__":
    main()
