#!/usr/bin/env python3
"""
features.py — the canonical 46-D theory-feature space for the second DL attempt.

Delegates entirely to the repo's feature modules (unsup_theory_features,
critical_scaling_features); the 51 -> 46 near-duplicate drop mirrors
paper_figures.build_features, which stays the single source of truth for the paper.
paper_figures itself is not imported here because its import pulls in the real-data
loaders; instead the drop set is restated and the resulting names are asserted to be 46.

Run directly to compute and cache features for the data.py outputs.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from unsup_theory_features import extract_theory_features, FEATURE_NAMES, TCFP_NAMES
from critical_scaling_features import extract_critical_features, CRIT_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "curated", "deep")

# mirror of paper_figures.py (the canonical definition) — keep in sync
_GROUP_FULL = (["A:CSD"] * 8 + ["B:Inflect"] * 7 + ["C:Phase"] * 6 + ["D:Catch22"] * 7
               + ["E:SN-fp"] * 5 + ["F:Transit"] * 5 + ["G:TC-fp"] * 5 + ["H:Crit"] * 8)
_ALL_FULL = list(FEATURE_NAMES) + list(TCFP_NAMES) + list(CRIT_NAMES)
DROP_FEATURES = {"peak_to_rms", "velocity_kurtosis", "center_fraction",
                 "percap_linearity", "fold_dwell"}
KEEP_IDX = [i for i, n in enumerate(_ALL_FULL) if n not in DROP_FEATURES]
FEAT_NAMES_46 = [_ALL_FULL[i] for i in KEEP_IDX]
FEAT_GROUPS_46 = [_GROUP_FULL[i] for i in KEEP_IDX]
assert len(FEAT_NAMES_46) == 46, f"expected 46 features, got {len(FEAT_NAMES_46)}"


def build_features46(X, verbose=False):
    """Canonical 46-D feature matrix for curves X (N, L). Same construction as
    paper_figures.build_features: 38 theory + 5 TC-fingerprint + 8 critical, minus the
    5 near-duplicates."""
    F = np.hstack([
        np.nan_to_num(extract_theory_features(X, verbose=verbose, include_tc_fingerprint=True)),
        np.nan_to_num(extract_critical_features(X)),
    ])
    return F[:, KEEP_IDX]


def main():
    for tag in ("matched", "natural"):
        X = np.load(f"{DATA}/X_{tag}.npy")
        t0 = time.time()
        F = build_features46(X)
        np.save(f"{DATA}/F_{tag}.npy", F.astype(np.float32))
        print(f"{tag}: features {F.shape} in {time.time()-t0:.0f}s -> F_{tag}.npy")


if __name__ == "__main__":
    main()
