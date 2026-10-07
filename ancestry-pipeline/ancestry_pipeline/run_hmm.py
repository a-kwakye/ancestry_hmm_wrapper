#!/usr/bin/env python3
"""Run 2- or 3-population Ancestry_HMM jobs in parallel and collect one posterior per sample."""
import argparse
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess


def run_sample(task):
    sample, panel, sample_file, outdir, executable, priors, pulses = task
    workdir = Path(outdir) / sample
    workdir.mkdir(parents=True, exist_ok=True)
    posterior_dir = Path(outdir) / "posteriors"
    posterior_dir.mkdir(parents=True, exist_ok=True)
    n_ancestries = len(priors)
    command = [executable, "-i", panel, "-s", sample_file, "-a", str(n_ancestries), *map(str, priors)]
    for ancestry, time, proportion in pulses:
        command += ["-p", str(ancestry), str(time), str(proportion)]
    command += ["-g"]
    with (workdir / "ancestry_hmm.log").open("w") as log:
        log.write(f"Arguments: {command!r}\n")
        log.flush()
        try:
            result = subprocess.run(command, cwd=workdir, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0:
                return sample, result.returncode, str(workdir)
            posteriors = sorted(p for p in workdir.iterdir() if p.is_file() and "posterior" in p.name.lower())
            if len(posteriors) != 1:
                log.write(f"Expected exactly one posterior, found {len(posteriors)}: {[p.name for p in posteriors]}\n")
                return sample, 1, str(workdir)
            destination = posterior_dir / f"{sample}.posterior"
            if destination.exists():
                destination.unlink()
            shutil.move(str(posteriors[0]), str(destination))
            log.write(f"Posterior saved: {destination}\n")
            return sample, 0, str(workdir)
        except OSError as error:
            log.write(f"Failed to run Ancestry_HMM or collect posterior: {error}\n")
            return sample, 127, str(workdir)


def parse_pulse(text):
    fields = text.split(",")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("pulse must be ANCESTRY,TIME,PROPORTION")
    try:
        ancestry = int(fields[0]); time = float(fields[1]); proportion = float(fields[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pulse must be ANCESTRY,TIME,PROPORTION") from exc
    if ancestry < 0:
        raise argparse.ArgumentTypeError("ancestry index must be nonnegative")
    if not 0 <= proportion <= 1:
        raise argparse.ArgumentTypeError("pulse proportion must be between 0 and 1")
    return ancestry, time, proportion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=".")
    parser.add_argument("--outdir", default="ancestry_hmm_results")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--executable", default="ancestry_hmm")
    parser.add_argument("--priors", nargs="+", type=float, required=True, metavar="P",
                        help="Two or three -a ancestry proportions in reference-index order")
    parser.add_argument("--pulse", action="append", type=parse_pulse, required=True,
                        help="Repeatable ANCESTRY_INDEX,TIME,PROPORTION")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if len(args.priors) not in (2, 3):
        parser.error("--priors must contain exactly 2 or 3 values")
    if any(x < 0 for x in args.priors) or abs(sum(args.priors) - 1.0) > 1e-6:
        parser.error("--priors must be nonnegative and sum to 1")
    if any(a >= len(args.priors) for a, _, _ in args.pulse):
        parser.error(f"pulse ancestry indices must be between 0 and {len(args.priors)-1}")
    executable = shutil.which(args.executable)
    if executable is None:
        parser.error(f"Executable not found: {args.executable}")
    executable = str(Path(executable).resolve())
    input_dir = Path(args.input_dir).resolve(); outdir = Path(args.outdir).resolve()
    panels = sorted(input_dir.glob("*.input.tsv"))
    if not panels:
        parser.error(f"No *.input.tsv files found in {input_dir}")
    tasks = []; missing = []
    for panel in panels:
        sample = panel.name.removesuffix(".input.tsv")
        sample_file = input_dir / f"{sample}.samples.txt"
        if not sample_file.is_file(): missing.append(str(sample_file))
        tasks.append((sample, str(panel), str(sample_file), str(outdir), executable, args.priors, args.pulse))
    if missing:
        parser.error("Missing sample files:\n" + "\n".join(missing))
    outdir.mkdir(parents=True, exist_ok=True)
    workers = min(args.jobs, len(tasks))
    print(f"Running {len(tasks)} samples with up to {workers} simultaneous jobs.", flush=True)
    print(f"Ancestries: {len(args.priors)}; priors: {args.priors}; pulses: {args.pulse}", flush=True)
    failed = []
    with (outdir / "run_summary.tsv").open("w") as summary:
        summary.write("sample\texit_code\toutput_directory\n")
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            for sample, code, directory in pool.imap_unordered(run_sample, tasks):
                summary.write(f"{sample}\t{code}\t{directory}\n"); summary.flush()
                print(f"{'OK' if code == 0 else 'FAILED'}: {sample} (exit {code})", flush=True)
                if code: failed.append(sample)
    if failed:
        print("Failed samples: " + ", ".join(failed), flush=True); raise SystemExit(1)
    print("All samples completed successfully.", flush=True)


if __name__ == "__main__":
    mp.freeze_support(); main()
