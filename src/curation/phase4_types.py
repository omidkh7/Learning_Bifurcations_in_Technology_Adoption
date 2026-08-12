# phase4_types.py

from dataclasses import dataclass
import numpy as np


@dataclass
class RealWorldSample:
    """
    One curated real-world technology adoption series.

    x_full is always exactly TARGET_POINTS (500) long — a PCHIP-interpolated
    uniform grid over the observed adoption phase [year_start, year_sat].
    No prefix cuts are stored; this class is used exclusively for
    unsupervised learning, not for the Phase 3/4 classifier pipeline.
    """
    # Identifiers
    row_id:        str
    tech_name:     str
    metric:        str
    unit:          str
    source:        str
    spatial_scale: str
    region:        str
    country:       str
    variable:      str

    # Calendar years of the adoption phase (raw, before resampling)
    years_full:    np.ndarray   # (n_observed,) int

    # Normalised adoption series — always TARGET_POINTS (500) long
    x_full:        np.ndarray   # (500,) float32

    # Diagnostics
    year_start:    int    # calendar year of first observed data point
    year_end:      int    # calendar year of last observed data point (full raw series)
    year_sat:      int    # calendar year of observed maximum (adoption phase end)
    series_length: int    # always TARGET_POINTS = 500
    n_observed:    int    # number of raw annual data points before resampling
    n_gaps_filled: int    # internal NaNs interpolated during gap-filling
    peak_idx:      int    # index of true maximum in the gap-filled raw series

    # Stage-1 Option-B saturation fields (added for full S-curve curation)
    series_type:       str   = "saturating"   # "saturating" | "pre_saturation" | "declining_structural"
    n_post_peak:       int   = 0              # raw annual post-peak points included (0 = peak-only)
    year_peak:         int   = 0              # calendar year of the observed maximum
    saturation_reached: bool = False          # True if X=5% / Y=3-yr saturation criterion was met
