#!/usr/bin/env python3
"""
adoption_clean.py — shared cleaning helpers for the adoption-shape / 38-D analyses.

Canonical rule (§56/§83): an adoption curve is the RISE + PLATEAU. Post-peak STRUCTURAL decline
(abandonment/substitution, e.g. a shut-down reactor) is NOT part of the adoption S-curve and must be
truncated before normalisation, else min-max ends the curve below 1 and the decline contaminates the
shape features (net_rise, monotonicity, inflection).
"""
import numpy as np


def truncate_decline(v, thr=0.90):
    """Drop the post-peak decline, keeping rise+plateau — only when the series ended below thr·peak."""
    v = np.asarray(v, float)
    if len(v) < 2:
        return v
    pk = int(v.argmax()); peak = v[pk]
    if peak > 0 and v[-1] < thr * peak and pk < len(v) - 1:
        end = pk
        while end + 1 < len(v) and v[end + 1] >= thr * peak:
            end += 1
        out = v[:end + 1]
        return out if len(out) >= 2 else v          # guard: never collapse below 2 points
    return v
