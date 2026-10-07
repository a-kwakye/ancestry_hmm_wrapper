#!/usr/bin/env python3
"""Split an existing Ancestry_HMM panel into one input per target sample.

The input panel must be headerless, with shared columns followed by TWO
allele-count columns per target, in exactly the order of --samples.
All markers, reference counts, recombination distances and optional shared
error-rate columns are copied without recalculation or sample-specific filtering.
Requires Python 3 only. Plain and .gz input panels are supported.

Example:
  python split_ancestry_hmm_per_sample.py \
    --panel ancestry_hmm_ET_FR_ZI_all_chroms/panel_all_chroms.tsv \
    --samples ancestry_hmm_ET_FR_ZI_all_chroms/samples_all_chroms.txt \
    --outdir ancestry_hmm_per_sample

Standard shared-column counts: 7 for two references, 9 for three references.
Otherwise inferred as panel width minus twice the number of target samples.
The samples file is the HMM ID/ploidy file, not a PLINK FID/IID keep file.
"""

import argparse
from contextlib import ExitStack
import csv
import gzip
from pathlib import Path
import re
import tempfile


def read_samples(path):
    samples = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected SAMPLE_ID PLOIDY")
        name, ploidy = fields
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
            raise ValueError(f"Unsafe output filename for sample {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate sample ID: {name}")
        if not ploidy.isdigit() or int(ploidy) < 1:
            raise ValueError(f"{name}: expected positive integer ploidy")
        samples.append((name, ploidy))
        seen.add(name)
    if not samples:
        raise ValueError("Samples file is empty")
    return samples


def open_panel(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return Path(path).open()


def split_panel(panel, sample_file, outdir, shared_columns=None, batch_size=64):
    panel = Path(panel).resolve()
    sample_file = Path(sample_file).resolve()
    outdir = Path(outdir).resolve()
    samples = read_samples(sample_file)
    with open_panel(panel) as source:
        first = next((line.split() for line in source if line.strip()), None)
    if first is None:
        raise ValueError("Panel is empty")
    width = len(first)
    inferred = width - 2 * len(samples)
    shared = inferred if shared_columns is None else shared_columns
    if shared < 5 or shared % 2 != 1 or shared != inferred:
        raise ValueError(
            f"Panel has {width} columns for {len(samples)} samples; expected an odd "
            f"shared prefix of at least 5 columns plus two columns per sample. "
            f"Inferred shared columns: {inferred}; requested: {shared_columns}. "
            "Check that the samples file matches the panel's target-column order."
        )
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    outdir.mkdir(parents=True, exist_ok=True)
    filenames = [f"{name}.{suffix}" for name, _ in samples
                 for suffix in ("input.tsv", "samples.txt")]
    filenames.append("manifest.tsv")
    for filename in filenames:
        destination = outdir / filename
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}; use a new output directory")
        if destination in (panel, sample_file):
            raise ValueError("An output path would replace an input")

    retained = None
    # Stage outputs so malformed rows do not leave apparently complete panels.
    # Limit open files; large cohorts use several streaming passes over the panel.
    with tempfile.TemporaryDirectory(prefix=".split_", dir=outdir) as temporary:
        stage = Path(temporary)
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            rows = 0
            with ExitStack() as stack:
                outputs = [stack.enter_context((stage / f"{name}.input.tsv").open("w"))
                           for name, _ in batch]
                source = stack.enter_context(open_panel(panel))
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    fields = line.split()
                    if len(fields) != width:
                        raise ValueError(f"{panel}:{line_number}: expected {width} columns, got {len(fields)}")
                    prefix = "\t".join(fields[:shared])
                    for offset, output in enumerate(outputs):
                        column = shared + 2 * (start + offset)
                        pair = fields[column:column + 2]
                        if not all(value.isdigit() for value in pair):
                            raise ValueError(f"{panel}:{line_number}: invalid allele counts for {batch[offset][0]}")
                        output.write(prefix + "\t" + "\t".join(pair) + "\n")
                    rows += 1
            if retained is not None and rows != retained:
                raise ValueError("Panel changed between streaming passes")
            retained = rows
            for name, ploidy in batch:
                (stage / f"{name}.samples.txt").write_text(f"{name}\t{ploidy}\n")
        with (stage / "manifest.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["sample", "ploidy", "input_file", "sample_file", "sites", "shared_columns"])
            for name, ploidy in samples:
                writer.writerow([name, ploidy, str(outdir / f"{name}.input.tsv"),
                                 str(outdir / f"{name}.samples.txt"), retained, shared])
        for filename in filenames:
            (stage / filename).rename(outdir / filename)
    print(f"Wrote {len(samples)} panels, each with {retained:,} markers and {shared + 2} columns.")
    print(f"Copied the same first {shared} columns to every panel; only the target allele-count pair changes.")
    print(f"Manifest: {outdir / 'manifest.tsv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", required=True, help="Existing headerless multi-sample HMM input")
    parser.add_argument("--samples", required=True, help="ID/ploidy file in the input panel's target order")
    parser.add_argument("--outdir", default="ancestry_hmm_per_sample")
    parser.add_argument("--shared-columns", type=int, help="Optional explicit count of fixed leading columns")
    parser.add_argument("--batch-size", type=int, default=64, help="Maximum simultaneously open panel outputs")
    args = parser.parse_args()
    try:
        split_panel(args.panel, args.samples, args.outdir, args.shared_columns, args.batch_size)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
