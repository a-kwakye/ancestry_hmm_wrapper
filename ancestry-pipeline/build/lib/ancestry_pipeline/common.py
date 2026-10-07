from itertools import combinations_with_replacement
from pathlib import Path


def parse_population_spec(text):
    if "=" not in text:
        raise ValueError("population must be NAME=KEEP_FILE")
    name, keep = text.split("=", 1)
    name = name.strip()
    keep = keep.strip()
    if not name or not keep or any(c.isspace() for c in name):
        raise ValueError("population must be NAME=KEEP_FILE with a nonempty name")
    return name, keep


def validate_population_names(names, minimum=2, maximum=3):
    if not (minimum <= len(names) <= maximum) or len(set(names)) != len(names):
        if minimum == maximum:
            raise ValueError(f"exactly {minimum} unique ancestry population names are required")
        raise ValueError(f"between {minimum} and {maximum} unique ancestry population names are required")


def diploid_states(n_populations):
    if n_populations not in (2, 3):
        raise ValueError("diploid posterior state handling supports 2 or 3 ancestry populations")
    states = []
    for i, j in combinations_with_replacement(range(n_populations), 2):
        counts = [0] * n_populations
        counts[i] += 1
        counts[j] += 1
        states.append(",".join(map(str, counts)))
    return states


def read_simple_samples(path):
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not rows:
        raise ValueError(f"{path}: sample list is empty")
    if all(len(row) == 1 for row in rows):
        ids = [row[0] for row in rows]
    elif all(len(row) == 2 and row[0] == "0" for row in rows):
        ids = [row[1] for row in rows]
    else:
        raise ValueError(f"{path}: expected SAMPLE_ID or 0 SAMPLE_ID on every row")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate sample ID")
    return ids


def read_population_map(path, allowed=None):
    mapping = {}
    if not path:
        return mapping
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if [field.lower() for field in fields] == ["sample", "population"]:
            continue
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected SAMPLE_ID POPULATION")
        sample, pop = fields
        if allowed is not None and pop not in allowed:
            raise ValueError(f"{path}:{line_number}: unknown population {pop!r}; expected one of {allowed}")
        if sample in mapping and mapping[sample] != pop:
            raise ValueError(f"{path}:{line_number}: conflicting population for {sample}")
        mapping[sample] = pop
    return mapping


def read_chromosome_map(path):
    mapping = {}
    order = []
    if not path:
        return mapping, order
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if [x.lower() for x in fields] in (["chrom", "label"], ["chromosome", "label"]):
            continue
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected CHROM LABEL")
        chrom, label = fields
        if chrom in mapping and mapping[chrom] != label:
            raise ValueError(f"{path}:{line_number}: conflicting label for chromosome {chrom}")
        mapping[chrom] = label
        if label not in order:
            order.append(label)
    return mapping, order
