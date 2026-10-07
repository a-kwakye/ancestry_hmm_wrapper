#!/usr/bin/env python3
"""Run the genome-agnostic two- or three-population Ancestry_HMM workflow end to end."""
import argparse
from pathlib import Path
import subprocess
import sys
import pandas as pd
from .common import parse_population_spec, read_population_map, validate_population_names


def run_module(module, args):
    command = [sys.executable, "-m", module, *map(str, args)]
    print("\n>>> " + " ".join(command), flush=True); subprocess.run(command, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vcf", required=True); p.add_argument("--map", required=True)
    p.add_argument("--population", action="append", required=True, metavar="NAME=KEEP_FILE", help="Repeat two or three times; order defines ancestry indices")
    p.add_argument("--target-samples"); p.add_argument("--population-map", help="SAMPLE_ID POPULATION table; required for distributions/painting")
    p.add_argument("--chromosomes", nargs="+"); p.add_argument("--chromosome-map")
    p.add_argument("--map-chrom-col", default="Arm"); p.add_argument("--map-start-col", default="Start"); p.add_argument("--map-stop-col", default="Stop"); p.add_argument("--map-rate-col", default="c (cM/Mb)")
    p.add_argument("--map-stop-exclusive", action="store_true"); p.add_argument("--rate-unit", choices=["cM/Mb", "M/bp"], default="cM/Mb")
    p.add_argument("--bp-space", type=int, default=1000); p.add_argument("--ploidy", type=int, default=2); p.add_argument("--ploidy-map")
    p.add_argument("--plink", default="plink"); p.add_argument("--jobs", type=int, default=10); p.add_argument("--executable", default="ancestry_hmm")
    p.add_argument("--priors", nargs="+", type=float, required=True, metavar="P", help="Exactly one prior per reference population")
    p.add_argument("--pulse", action="append", required=True); p.add_argument("--state-order", nargs="+", required=True, metavar="POP")
    p.add_argument("--weighting", choices=["length", "snps"], default="length"); p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--distribution-groups", nargs="+"); p.add_argument("--outdir", default="ancestry_analysis")
    p.add_argument("--skip-distributions", action="store_true"); p.add_argument("--skip-painting", action="store_true")
    args = p.parse_args()
    try:
        pop_specs = [parse_population_spec(x) for x in args.population]; reference_names = [x[0] for x in pop_specs]
        validate_population_names(reference_names); validate_population_names(args.state_order)
    except ValueError as exc: p.error(str(exc))
    if len(args.priors) != len(reference_names): p.error("--priors must contain exactly one value per --population")
    if len(args.state_order) != len(reference_names) or set(reference_names) != set(args.state_order): p.error("--state-order must contain the same names supplied with --population")
    if (not args.skip_distributions or not args.skip_painting) and not args.population_map: p.error("--population-map is required unless both plotting stages are skipped")
    root = Path(args.outdir).resolve(); panel_dir=root/"panel"; split_dir=root/"per_sample"; hmm_dir=root/"hmm"; summary_dir=root/"summaries"; plot_dir=root/"plots"; paint_dir=plot_dir/"chromosome_paintings"
    for d in [panel_dir, split_dir, hmm_dir, summary_dir, plot_dir, paint_dir]: d.mkdir(parents=True, exist_ok=True)
    build_args=["--vcf",args.vcf,"--map",args.map,"--outdir",panel_dir,"--bp-space",args.bp_space,"--ploidy",args.ploidy,"--plink",args.plink,"--map-chrom-col",args.map_chrom_col,"--map-start-col",args.map_start_col,"--map-stop-col",args.map_stop_col,"--map-rate-col",args.map_rate_col,"--rate-unit",args.rate_unit]
    for spec in args.population: build_args += ["--population",spec]
    if args.target_samples: build_args += ["--target-samples",args.target_samples]
    if args.population_map: build_args += ["--population-map",args.population_map]
    if args.ploidy_map: build_args += ["--ploidy-map",args.ploidy_map]
    if args.chromosomes: build_args += ["--chromosomes",*args.chromosomes]
    if args.map_stop_exclusive: build_args += ["--map-stop-exclusive"]
    run_module("ancestry_pipeline.build_panel",build_args)
    panel=panel_dir/"panel_all_chroms.tsv"; samples=panel_dir/"samples_all_chroms.txt"
    run_module("ancestry_pipeline.split_samples",["--panel",panel,"--samples",samples,"--outdir",split_dir])
    hmm_args=["--input-dir",split_dir,"--outdir",hmm_dir,"--jobs",args.jobs,"--executable",args.executable,"--priors",*args.priors]
    for pulse in args.pulse: hmm_args += ["--pulse",pulse]
    run_module("ancestry_pipeline.run_hmm",hmm_args)
    summary=summary_dir/"ancestry_summary.tsv"
    run_module("ancestry_pipeline.summarize",["--folder",hmm_dir/"posteriors","--out",summary,"--weighting",args.weighting,"--state-order",*args.state_order])
    if not args.skip_distributions:
        dist_args=["--summary",summary,"--population-map",args.population_map,"--state-order",*args.state_order,"--out-prefix",plot_dir/"ancestry_distributions"]
        if args.distribution_groups: dist_args += ["--groups",*args.distribution_groups]
        run_module("ancestry_pipeline.distributions",dist_args)
    if not args.skip_painting:
        mapping=read_population_map(args.population_map); data=pd.read_csv(summary,sep="\t",dtype={"sample":str}); data["sample_group"]=data["sample"].map(mapping)
        for population in args.state_order:
            if population not in set(data["sample_group"].dropna()): print(f"Skipping paintings for {population}: no sample group with that name in population map.",flush=True); continue
            for group in ["high","low"]:
                mask=data["sample_group"].eq(population); mask &= data[population].gt(args.threshold) if group=="high" else data[population].lt(args.threshold)
                if not mask.any(): print(f"Skipping {group} {population} painting: no matching samples.",flush=True); continue
                output=paint_dir/f"{group}_{population}_chromosome_paintings.pdf"
                paint_args=["--population",population,"--sample-group",population,"--group",group,"--summary",summary,"--posterior-dir",hmm_dir/"posteriors","--population-map",args.population_map,"--state-order",*args.state_order,"--threshold",args.threshold,"--out",output]
                if args.chromosome_map: paint_args += ["--chromosome-map",args.chromosome_map]
                run_module("ancestry_pipeline.paint",paint_args)
    print(f"\nPipeline complete: {root}\nSummary: {summary}\nPosteriors: {hmm_dir/'posteriors'}\nPlots: {plot_dir}")


if __name__ == "__main__": main()
