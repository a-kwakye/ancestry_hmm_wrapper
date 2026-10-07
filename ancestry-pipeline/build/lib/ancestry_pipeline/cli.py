#!/usr/bin/env python3
"""Unified CLI for ancestry-hmm-pipeline."""
import subprocess
import sys
COMMANDS={"build":"ancestry_pipeline.build_panel","split":"ancestry_pipeline.split_samples","run-hmm":"ancestry_pipeline.run_hmm","summarize":"ancestry_pipeline.summarize","distributions":"ancestry_pipeline.distributions","paint":"ancestry_pipeline.paint","run":"ancestry_pipeline.workflow"}
HELP="""ancestry-pipeline v0.2.1: genome-agnostic 2- or 3-population Ancestry_HMM workflow\n\nCommands:\n  build           Build a 2- or 3-reference multi-sample panel\n  split           Split a panel into one HMM input per target\n  run-hmm         Run Ancestry_HMM in parallel\n  summarize       Summarize posterior ancestry proportions\n  distributions   Compare ancestry distributions across sample groups\n  paint           Paint chromosome-level ancestry for selected samples\n  run             Execute the complete workflow\n\nUse 'ancestry-pipeline COMMAND --help' for command-specific options.\n"""
def main():
    if len(sys.argv)<2 or sys.argv[1] in {"-h","--help"}: print(HELP); return
    command=sys.argv[1]
    if command not in COMMANDS: print(f"Unknown command: {command}\n\n{HELP}",file=sys.stderr); raise SystemExit(2)
    raise SystemExit(subprocess.call([sys.executable,"-m",COMMANDS[command],*sys.argv[2:]]))
if __name__=="__main__": main()
