#!/bin/bash
# Proper (reviewer-proofing) DL run:
#   benchmark_dl.py proper            10,000 series/class, 10 model seeds, 3 benchmark draws
#   confidence_distribution.py proper 10,000 series/class, 10 model seeds
# Writes *_proper outputs only (runs/benchmark/benchmark_dl_proper.json + fig_benchmark_dl_proper.png,
# runs/checks/confidence_distribution_proper.json + fig_confidence_proper.png); the canonical
# files are left untouched until the results are adopted into the manuscript.
#
# Detached launch (survives terminal close; caffeinate -is keeps the Mac awake on AC power):
#   cd <repo root>
#   nohup caffeinate -is bash second_attempt_deep/proper_dl_run.sh > second_attempt_deep/runs/proper_dl.log 2>&1 &
set -u
cd "$(dirname "$0")/.."       # repo root: all data paths are relative to it
echo "=== PROPER DL RUN START $(date) ==="
python second_attempt_deep/benchmark_dl.py proper
echo "--- benchmark_dl proper exit=$? $(date) ---"
python second_attempt_deep/confidence_distribution.py proper
echo "--- confidence_distribution proper exit=$? $(date) ---"
echo "=== PROPER DL RUN DONE $(date) ==="
