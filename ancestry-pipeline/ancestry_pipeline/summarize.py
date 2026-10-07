#!/usr/bin/env python3
"""Summarize diploid 2- or 3-ancestry posterior files with arbitrary population names."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .common import diploid_states, validate_population_names


def summarize(path, weighting, state_order):
    validate_population_names(state_order)
    states = diploid_states(len(state_order))
    copy_fractions = np.array([[int(x) for x in state.split(",")] for state in states], dtype=float) / 2
    data = pd.read_csv(path, sep=r"\s+", dtype={"chrom": str})
    required = ["chrom", "position", *states]
    if not set(required).issubset(data.columns):
        raise ValueError(f"{path}: expected posterior state columns {states}")
    if data.empty or data[required].isna().any().any():
        raise ValueError(f"{path}: empty data or missing values")
    posterior = data[states].to_numpy(dtype=float)
    totals = posterior.sum(axis=1)
    if not np.isfinite(posterior).all() or (posterior < 0).any() or not np.allclose(totals, 1, atol=0.001, rtol=0):
        raise ValueError(f"{path}: invalid posterior probabilities or row sums")
    posterior = posterior / totals[:, None]
    ancestry = posterior @ copy_fractions
    positions = pd.to_numeric(data["position"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(positions).all() or (positions < 1).any() or (positions != np.floor(positions)).any():
        raise ValueError(f"{path}: invalid positions")
    weights = np.zeros(len(data), dtype=float); covered_bp = 0.0; singleton_segments = 0
    for chrom, indices in data.groupby("chrom", sort=False).indices.items():
        gaps = np.diff(positions[indices])
        if (gaps <= 0).any(): raise ValueError(f"{path}: duplicate or unsorted positions on {chrom}")
        if len(indices) == 1:
            singleton_segments += 1; continue
        weights[indices[:-1]] += gaps / 2; weights[indices[1:]] += gaps / 2; covered_bp += float(gaps.sum())
    if weighting == "snps": weights[:] = 1
    if weights.sum() <= 0: raise ValueError(f"{path}: no marker spans to length-weight; use --weighting snps")
    proportions = dict(zip(state_order, np.average(ancestry, axis=0, weights=weights)))
    row = {"sample": path.name.removesuffix(".posterior")}
    row.update({name: proportions[name] for name in state_order})
    row.update({"n_sites": len(data), "n_segments": data["chrom"].nunique(), "covered_bp": int(covered_bp),
                "singleton_segments": singleton_segments, "weighting": weighting, "state_order": ",".join(state_order),
                "n_ancestries": len(state_order)})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="ancestry_hmm_results/posteriors")
    parser.add_argument("--out")
    parser.add_argument("--weighting", choices=["length", "snps"], default="length")
    parser.add_argument("--state-order", nargs="+", required=True, metavar="POP",
                        help="Two or three population names corresponding to posterior state components")
    args = parser.parse_args()
    try: validate_population_names(args.state_order)
    except ValueError as exc: parser.error(str(exc))
    folder = Path(args.folder); paths = sorted(folder.glob("*.posterior"))
    if not paths: parser.error(f"No *.posterior files found in {folder}")
    rows = []
    for path in paths:
        try: rows.append(summarize(path, args.weighting, args.state_order))
        except (ValueError, OSError) as exc: parser.error(str(exc))
    result = pd.DataFrame(rows)
    if result["sample"].duplicated().any(): parser.error("Duplicate sample files found; keep one posterior per sample")
    output = Path(args.out) if args.out else folder / "ancestry_summary.tsv"
    output.parent.mkdir(parents=True, exist_ok=True); result.to_csv(output, sep="\t", index=False, float_format="%.10g")
    print(f"Saved {len(rows)} samples to {output}")
    print(result[["sample", *args.state_order]].head().to_string(index=False))


if __name__ == "__main__": main()
