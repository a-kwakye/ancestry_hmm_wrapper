from ancestry_pipeline.common import diploid_states, parse_population_spec, read_chromosome_map, validate_population_names
from ancestry_pipeline.summarize import summarize


def test_population_spec_and_names():
    assert parse_population_spec("A=a.keep") == ("A", "a.keep")
    validate_population_names(["A", "B"])
    validate_population_names(["A", "B", "C"])


def test_states():
    assert diploid_states(2) == ["2,0", "1,1", "0,2"]
    assert diploid_states(3) == ["2,0,0", "1,1,0", "1,0,1", "0,2,0", "0,1,1", "0,0,2"]


def test_chromosome_map(tmp_path):
    p=tmp_path/"chrom.tsv"; p.write_text("chrom label\n1 chr1\nX chrX\n")
    mapping,order=read_chromosome_map(p); assert mapping=={"1":"chr1","X":"chrX"}; assert order==["chr1","chrX"]


def test_three_pop_summary(tmp_path):
    p=tmp_path/"sample.posterior"; p.write_text("chrom position 2,0,0 1,1,0 1,0,1 0,2,0 0,1,1 0,0,2\n1 100 1 0 0 0 0 0\n1 200 0 0 0 1 0 0\n")
    row=summarize(p,"snps",["Lake","Coastal","River"]); assert abs(row["Lake"]-.5)<1e-12; assert abs(row["Coastal"]-.5)<1e-12


def test_two_pop_summary(tmp_path):
    p=tmp_path/"sample.posterior"; p.write_text("chrom position 2,0 1,1 0,2\n1 100 1 0 0\n1 200 0 1 0\n")
    row=summarize(p,"snps",["ZI","FR"]); assert abs(row["ZI"]-.75)<1e-12; assert abs(row["FR"]-.25)<1e-12; assert row["n_ancestries"]==2
