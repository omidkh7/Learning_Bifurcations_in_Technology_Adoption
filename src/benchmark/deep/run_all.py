#!/usr/bin/env python3
"""Run the full second attempt end to end: data -> features -> train -> evaluate."""
import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

for step in ("data.py", "features.py", "train.py", "evaluate.py"):
    print(f"\n{'='*70}\nRUN {step}\n{'='*70}")
    r = subprocess.run([sys.executable, os.path.join(HERE, step)], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"{step} failed with code {r.returncode}")
print("\nAll steps complete. See second_attempt_deep/runs/RESULTS.md")
