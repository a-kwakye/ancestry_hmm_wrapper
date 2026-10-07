#!/usr/bin/env python3
"""Build a genome-agnostic diploid two- or three-reference Ancestry_HMM panel.

Two or three --population NAME=KEEP_FILE arguments define ancestry indices 0, 1,
and optionally 2 in the supplied order. No population is inferred from sample names.
Targets are read from --target-samples; if omitted, every VCF sample is tested.
The recombination-map column names and chromosome set are configurable.
"""

import argparse
import csv
import gzip
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .common import parse_population_spec, read_population_map, read_simple_samples, validate_population_names


def read_vcf_samples(path):
    opener = gzip.open if str(path).endswith((".gz", ".bgz")) else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("#CHROM\t"):
                samples = line.rstrip("\r\n").split("\t")[9:]
                if not samples or len(set(samples)) != len(samples):
                    raise ValueError("VCF must contain nonempty, unique sample IDs")
                return samples
    raise ValueError(f"{path}: no #CHROM header found")


def make_maps(path, chrom_col, start_col, stop_col, rate_col, stop_inclusive=True, rate_unit="cM/Mb"):
    data = pd.read_csv(path, sep=None, engine="python", dtype={chrom_col: str})
    data.columns = data.columns.str.strip()
    required = [chrom_col, start_col, stop_col, rate_col]
    if not set(required).issubset(data.columns):
        raise ValueError(f"Map must have columns {required}; got {list(data.columns)}")
    work = data[required].copy()
    work[chrom_col] = work[chrom_col].astype(str)
    for col in (start_col, stop_col, rate_col):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna()
    if not np.all(work[[start_col, stop_col]].to_numpy() == np.floor(work[[start_col, stop_col]].to_numpy())):
        raise ValueError("Map start/stop coordinates must be integers")
    work = work.sort_values([chrom_col, start_col], kind="stable")
    scale = 1e-8 if rate_unit == "cM/Mb" else 1.0
    maps = {}
    for chrom, group in work.groupby(chrom_col, sort=False):
        start = group[start_col].to_numpy(dtype=np.int64)
        end = group[stop_col].to_numpy(dtype=np.int64) + (1 if stop_inclusive else 0)
        rate = group[rate_col].to_numpy(dtype=float)
        if (np.any(start < 0) or np.any(end <= start) or np.any(start[1:] < end[:-1]) or
                np.any(~np.isfinite(rate)) or np.any(rate < 0)):
            raise ValueError(f"Invalid or overlapping map intervals on chromosome {chrom}")
        length = end - start
        genetic_prefix = np.r_[0.0, np.cumsum(length * rate * scale)]
        covered_prefix = np.r_[0, np.cumsum(length)]
        maps[str(chrom)] = (start, end, rate * scale, genetic_prefix, covered_prefix)
    return maps


def integrate(x, m):
    start, end, rate_m_per_bp, genetic_prefix, covered_prefix = m
    idx = np.searchsorted(start, x, side="right") - 1
    safe = np.maximum(idx, 0)
    partial = np.clip(x - start[safe], 0, end[safe] - start[safe])
    genetic = np.where(idx >= 0, genetic_prefix[safe] + partial * rate_m_per_bp[safe], 0)
    covered = np.where(idx >= 0, covered_prefix[safe] + partial, 0)
    return genetic, covered


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--population", action="append", required=True, metavar="NAME=KEEP_FILE",
                    help="Reference population; repeat two or three times. Order defines ancestry indices 0,1[,2].")
    ap.add_argument("--target-samples", help="Optional sample list to analyze. Default: all VCF samples.")
    ap.add_argument("--population-map", help="Optional SAMPLE_ID POPULATION table recorded/validated for downstream grouping.")
    ap.add_argument("--outdir", default="ancestry_hmm_panel")
    ap.add_argument("--bp-space", type=int, default=1000)
    ap.add_argument("--ploidy", type=int, default=2, help="Default ploidy written for targets (default: 2)")
    ap.add_argument("--ploidy-map", help="Optional SAMPLE_ID PLOIDY table overriding --ploidy per target")
    ap.add_argument("--chromosomes", nargs="+", help="Optional chromosome labels to pass to PLINK and retain. Default: all chromosomes present in map/VCF.")
    ap.add_argument("--map-chrom-col", default="Arm")
    ap.add_argument("--map-start-col", default="Start")
    ap.add_argument("--map-stop-col", default="Stop")
    ap.add_argument("--map-rate-col", default="c (cM/Mb)")
    ap.add_argument("--map-stop-exclusive", action="store_true", help="Treat recombination-map stop coordinate as exclusive (default: inclusive)")
    ap.add_argument("--rate-unit", choices=["cM/Mb", "M/bp"], default="cM/Mb")
    ap.add_argument("--plink", default="plink", help="PLINK 1.9 executable name or path")
    args = ap.parse_args()

    if args.bp_space < 1 or args.ploidy < 1:
        ap.error("--bp-space and --ploidy must be positive")
    try:
        population_specs = [parse_population_spec(x) for x in args.population]
        population_names = [x[0] for x in population_specs]
        validate_population_names(population_names)
        reference_lists = [read_simple_samples(x[1]) for x in population_specs]
        population_map = read_population_map(args.population_map, allowed=population_names) if args.population_map else {}
    except ValueError as exc:
        ap.error(str(exc))

    all_samples = read_vcf_samples(args.vcf)
    all_set = set(all_samples)
    for name, refs in zip(population_names, reference_lists):
        missing = sorted(set(refs) - all_set)
        if missing:
            ap.error(f"Reference population {name} has samples absent from VCF: {missing}")
        if not refs:
            ap.error(f"Reference population {name} is empty")
    ref_membership = {}
    for name, refs in zip(population_names, reference_lists):
        for sample in refs:
            if sample in ref_membership and ref_membership[sample] != name:
                ap.error(f"Reference sample {sample} appears in both {ref_membership[sample]} and {name}")
            ref_membership[sample] = name
    for sample, pop in population_map.items():
        if sample in ref_membership and ref_membership[sample] != pop:
            ap.error(f"population map conflicts with reference keep list for {sample}")

    if args.target_samples:
        try:
            targets = read_simple_samples(args.target_samples)
        except ValueError as exc:
            ap.error(str(exc))
        missing = sorted(set(targets) - all_set)
        if missing:
            ap.error(f"Target samples absent from VCF: {missing}")
        target_set = set(targets)
        targets = [s for s in all_samples if s in target_set]
    else:
        targets = list(all_samples)

    maps = make_maps(args.map, args.map_chrom_col, args.map_start_col, args.map_stop_col,
                     args.map_rate_col, not args.map_stop_exclusive, args.rate_unit)
    selected = [str(c) for c in args.chromosomes] if args.chromosomes else list(maps)
    missing_map_chrom = [c for c in selected if c not in maps]
    if missing_map_chrom:
        ap.error(f"Requested chromosomes absent from recombination map: {missing_map_chrom}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "population_order.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["ancestry_index", "population", "keep_file"])
        for i, (name, keep_file) in enumerate(population_specs):
            writer.writerow([i, name, keep_file])
    with (outdir / "sample_membership.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample", "population", "reference_index", "is_target"])
        for sample in all_samples:
            pop = ref_membership.get(sample, population_map.get(sample, ""))
            idx = population_names.index(pop) if pop in population_names else ""
            writer.writerow([sample, pop, idx, int(sample in set(targets))])

    keep = outdir / "all_samples.keep"
    with keep.open("w") as handle:
        for sample in all_samples:
            handle.write(f"0\t{sample}\n")

    prefix = outdir / "genotypes"
    command = [args.plink, "--vcf", args.vcf, "--const-fid", "0", "--keep", str(keep)]
    if args.chromosomes:
        command += ["--chr", *selected]
    command += ["--biallelic-only", "strict", "--snps-only", "just-acgt", "--vcf-half-call", "missing",
                "--keep-allele-order", "--bp-space", str(args.bp_space), "--make-bed", "--out", str(prefix)]
    print("Importing VCF with PLINK and thinning sites...", flush=True)
    subprocess.run(command, check=True)
    subprocess.run([args.plink, "--bfile", str(prefix), "--keep-allele-order", "--recode", "A-transpose", "--out", str(prefix)], check=True)

    ploidy_overrides = {}
    if args.ploidy_map:
        for line_number, line in enumerate(Path(args.ploidy_map).read_text().splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if [x.lower() for x in fields] == ["sample", "ploidy"]:
                continue
            if len(fields) != 2 or not fields[1].isdigit() or int(fields[1]) < 1:
                ap.error(f"{args.ploidy_map}:{line_number}: expected SAMPLE_ID positive_integer_PLOIDY")
            ploidy_overrides[fields[0]] = int(fields[1])
        extra = sorted(set(ploidy_overrides) - set(targets))
        if extra:
            ap.error(f"Ploidy map contains non-target samples: {extra}")

    output = outdir / "panel_all_chroms.tsv"
    sample_output = outdir / "samples_all_chroms.txt"
    with sample_output.open("w") as handle:
        for sample in targets:
            handle.write(f"{sample}\t{ploidy_overrides.get(sample, args.ploidy)}\n")

    reference_cols = [[f"0_{s}" for s in group] for group in reference_lists]
    target_cols = [f"0_{s}" for s in targets]
    required_cols = ["CHR", "POS", *[col for group in reference_cols for col in group], *target_cols]
    missing_map = missing_refs = retained = 0
    previous = {}
    segment_number = {}

    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for chunk in pd.read_csv(str(prefix) + ".traw", sep="\t", chunksize=10000, dtype={"CHR": str}):
            if not set(required_cols).issubset(chunk.columns):
                missing = sorted(set(required_cols) - set(chunk.columns))
                raise ValueError(f"VCF/PLINK missing samples or columns: {missing}")
            for _, record in chunk.iterrows():
                chrom = str(record["CHR"])
                if chrom not in maps or chrom not in selected:
                    continue
                pos = int(record["POS"])
                m = maps[chrom]
                start, end = m[:2]
                i = np.searchsorted(start, pos, side="right") - 1
                if i < 0 or pos >= end[i]:
                    missing_map += 1
                    continue
                if chrom in previous and pos <= previous[chrom][0]:
                    raise ValueError(f"Sites out of order or duplicated on chromosome {chrom}: {pos}")

                def pair(columns):
                    dosage = pd.to_numeric(record[columns], errors="coerce").to_numpy(dtype=float)
                    if np.any(~np.isnan(dosage) & ~np.isin(dosage, [0, 1, 2])):
                        raise ValueError(f"Non-diploid/invalid genotype dosage on {chrom}:{pos}")
                    called = np.isfinite(dosage)
                    a = np.where(called, dosage, 0).astype(int)
                    b = np.where(called, 2 - dosage, 0).astype(int)
                    return a, b

                reference_counts = []
                for columns in reference_cols:
                    ref_a, ref_b = pair(columns)
                    reference_counts.append((int(ref_a.sum()), int(ref_b.sum())))
                if any(a_count + b_count == 0 for a_count, b_count in reference_counts):
                    missing_refs += 1
                    continue
                a, b = pair(target_cols)
                alleles = np.column_stack((a, b)).reshape(-1)
                gen, covered = integrate(np.array([pos]), m)
                if chrom in previous:
                    old_pos, old_gen, old_covered = previous[chrom]
                    if covered[0] - old_covered != pos - old_pos:
                        segment_number[chrom] = segment_number.get(chrom, 1) + 1
                        distance = 0.0
                    else:
                        distance = float(gen[0] - old_gen)
                else:
                    distance = 0.0
                segment = segment_number.get(chrom, 1)
                output_chrom = chrom if segment == 1 else f"{chrom}_seg{segment}"
                writer.writerow([output_chrom, pos, *[c for counts in reference_counts for c in counts], distance, *alleles.tolist()])
                previous[chrom] = (pos, gen[0], covered[0])
                retained += 1

    if not retained:
        raise ValueError("No sites retained")
    print("Population order: " + ", ".join(f"{i}={p}" for i, p in enumerate(population_names)))
    print(f"Saved {retained:,} sites and {len(targets)} targets to {output}")
    print(f"Excluded: {missing_map:,} sites outside map; {missing_refs:,} without calls in one or more reference populations")
    print(f"Sample file: {sample_output}")


if __name__ == "__main__":
    main()
