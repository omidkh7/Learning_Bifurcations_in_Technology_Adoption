#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
curate_from_raw.py — standalone data curation for
"Learning Bifurcations in Technology Adoption" (SI Section S1).

ONE self-contained file that regenerates every curated tier described in SI S1
from the two raw HATCH exports. It combines the logic of Real_World_Curator.py,
World_Data_Curator.py, the merge/dedup/save of Combined_Dataset_Analysis.py,
create_filtered_dataset.py, create_genuine_v2_dataset.py, and plot_scurve_quality.py.

It has NO dependency on phase4_types.py / unsup_theory_features.py / unsup_real_world.py:
the RealWorldSample container is inlined, and the GMM / theory-feature analysis
(which is NOT part of curation) is omitted.

Inputs  (relative to the repo root; this file is expected to live in data/raw/):
    data/raw/technologies.csv     HATCH regional export
    data/raw/tech_world.csv       HATCH global export

Outputs:
    data/curated/real_world/              4,124  regional curated series
    data/curated/world_data/                137  global curated series
    data/curated/combined/                4,217  merged + de-duplicated
    data/curated/combined_filter10/       3,540  start < 10% of ceiling
    data/curated/combined_genuine/        1,658  genuine_v1 (drop non-adoption categories)
    data/curated/combined_genuine_v2/     1,372  genuine_v2 (drop 7 further categories) — MAIN
    results/unsup/shape_diagnostic/scurve_quality.csv
                                          s-score → hardexcl 815 / s002 327 / s005 224

Each curated directory holds: X_full.npy (N,500) float32, lengths.npy, metadata.csv,
real_world_samples.pkl (list of RealWorldSample), filter_info.json.

NOTE on genuine_v1: the original create_genuine_dataset.py was unavailable when this file
was assembled, so its exact tech-name removal list was recovered from the committed
combined/combined_genuine metadata and embedded below as GENUINE_V1_REMOVE (108 names:
vaccines, raw-material / industrial production, food/agriculture, mining/refining). Removing
these names from `combined` reproduces combined_genuine (1,658) exactly.

Usage:
    python data/raw/curate_from_raw.py
"""

import json
import os
import pickle
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# ── Paths (this file lives in <root>/data/raw/) ───────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(os.path.dirname(_HERE))          # <root>/data/raw -> <root>
RAW      = os.path.join(ROOT, "data", "raw")
CURATED  = os.path.join(ROOT, "data", "curated")
SHAPEOUT = os.path.join(ROOT, "results", "unsup", "shape_diagnostic")

TECH_REGIONAL = os.path.join(RAW, "technologies.csv")
TECH_WORLD    = os.path.join(RAW, "tech_world.csv")


# =============================================================================
# 0.  Inlined sample container (was phase4_types.RealWorldSample)
# =============================================================================
@dataclass
class RealWorldSample:
    """One curated real-world technology adoption series (500-pt PCHIP grid)."""
    row_id:        str
    tech_name:     str
    metric:        str
    unit:          str
    source:        str
    spatial_scale: str
    region:        str
    country:       str
    variable:      str
    years_full:    np.ndarray
    x_full:        np.ndarray
    year_start:    int
    year_end:      int
    year_sat:      int
    series_length: int
    n_observed:    int
    n_gaps_filled: int
    peak_idx:      int
    series_type:        str  = "saturating"
    n_post_peak:        int  = 0
    year_peak:          int  = 0
    saturation_reached: bool = False


# =============================================================================
# 1.  Curation constants (Real_World_Curator.py)
# =============================================================================
METADATA_COLS = [
    "ID", "Spatial Scale", "Region", "Country Name",
    "Technology Name", "Metric", "Unit", "Data Source", "Variable",
]
MIN_YEARS      = 10
MIN_VARIATION  = 0.05
TARGET_POINTS  = 500
MAX_GAP        = 5
SAT_THRESHOLD  = 0.05
SAT_DURATION   = 3
DECLINE_THRESHOLD = 0.15
DECLINE_DURATION  = 5

DISRUPTION_WINDOWS = [
    (1914, 1919), (1917, 1922), (1929, 1934), (1939, 1946), (1950, 1953),
    (1956, 1957), (1959, 1962), (1965, 1975), (1968, 1968), (1973, 1975),
    (1979, 1982), (1980, 1988), (1984, 1985), (1989, 1992), (1990, 1991),
    (1994, 1994), (1997, 1999), (2001, 2002), (2002, 2003), (2007, 2009),
    (2010, 2012), (2011, 2011), (2014, 2016), (2015, 2015), (2020, 2022),
    (2022, 2024),
]


@dataclass
class RejectedSeries:
    row_id:    str
    tech_name: str
    reason:    str


# =============================================================================
# 2.  Curation helpers (Real_World_Curator.py)
# =============================================================================
def interpolate_gaps(x: np.ndarray, max_gap: int = MAX_GAP):
    x = x.copy().astype(float)
    n_filled = 0
    i = 0
    while i < len(x):
        if np.isnan(x[i]):
            j = i
            while j < len(x) and np.isnan(x[j]):
                j += 1
            gap_len = j - i
            if gap_len <= max_gap and i > 0 and j < len(x):
                x[i:j] = np.linspace(x[i - 1], x[j], gap_len + 2)[1:-1]
                n_filled += gap_len
            i = j
        else:
            i += 1
    return x, n_filled


def normalize_series(x: np.ndarray) -> np.ndarray:
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if xmax - xmin < 1e-10:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)


def pchip_interpolate_to_target(x_adoption: np.ndarray, target: int = TARGET_POINTS) -> np.ndarray:
    n = len(x_adoption)
    if n == target:
        return x_adoption.astype(np.float32)
    t_orig = np.linspace(0.0, 1.0, n)
    t_new  = np.linspace(0.0, 1.0, target)
    interp = PchipInterpolator(t_orig, x_adoption, extrapolate=False)
    out    = interp(t_new)
    out[0]  = x_adoption[0]
    out[-1] = x_adoption[-1]
    nan_mask = np.isnan(out)
    if nan_mask.any():
        idx = np.arange(target)
        out[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], out[~nan_mask])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _in_disruption_window(year_start: int, year_end: int, windows=DISRUPTION_WINDOWS) -> bool:
    for ws, we in windows:
        if year_start <= we and year_end >= ws:
            return True
    return False


def detect_saturation(x_post, years_post, peak_val, threshold=SAT_THRESHOLD, duration=SAT_DURATION):
    within = np.abs(x_post - peak_val) <= threshold * (abs(peak_val) + 1e-10)
    n = len(within)
    for i in range(n - duration + 1):
        if within[i: i + duration].all():
            end = i + duration - 1
            while end + 1 < n and within[end + 1]:
                end += 1
            return end
    return None


def detect_structural_decline(x_post, years_post, peak_val,
                              dec_threshold=DECLINE_THRESHOLD, dec_duration=DECLINE_DURATION,
                              disruption_wins=DISRUPTION_WINDOWS):
    below = (peak_val - x_post) / (abs(peak_val) + 1e-10) > dec_threshold
    n = len(below)
    for i in range(n - dec_duration + 1):
        if below[i: i + dec_duration].all():
            yr_s = int(years_post[i]); yr_e = int(years_post[i + dec_duration - 1])
            if not _in_disruption_window(yr_s, yr_e, disruption_wins):
                return True
    return False


# =============================================================================
# 3.  The curator (Real_World_Curator.py) — Stage 1 / Option B full S-curve
# =============================================================================
class RealWorldCurator:
    def __init__(self, filepath, metadata_cols=None, min_years=MIN_YEARS,
                 min_variation=MIN_VARIATION, max_gap=MAX_GAP,
                 sat_threshold=SAT_THRESHOLD, sat_duration=SAT_DURATION,
                 decline_threshold=DECLINE_THRESHOLD, decline_duration=DECLINE_DURATION):
        self.filepath = filepath
        self.metadata_cols = metadata_cols or METADATA_COLS
        self.min_years = min_years
        self.min_variation = min_variation
        self.max_gap = max_gap
        self.sat_threshold = sat_threshold
        self.sat_duration = sat_duration
        self.decline_threshold = decline_threshold
        self.decline_duration = decline_duration
        self.samples = []
        self.rejected = []
        self._df = None
        self._year_cols = []

    def load(self, verbose=True):
        ext = os.path.splitext(self.filepath)[1].lower()
        df = pd.read_excel(self.filepath, dtype=str) if ext in (".xls", ".xlsx") \
            else pd.read_csv(self.filepath, dtype=str)
        year_cols = []
        for col in df.columns:
            try:
                yr = int(str(col).strip())
                if 1500 <= yr <= 2100:
                    year_cols.append(yr)
            except ValueError:
                pass
        for yr in year_cols:
            df[yr] = pd.to_numeric(df[str(yr) if str(yr) in df.columns else yr], errors="coerce")
        col_map = {}
        for expected in self.metadata_cols:
            for actual in df.columns:
                if str(actual).strip().lower() == expected.lower():
                    col_map[actual] = expected
        df = df.rename(columns=col_map)
        for col in self.metadata_cols:
            if col not in df.columns:
                df[col] = ""
        self._df = df
        self._year_cols = sorted(year_cols)
        if verbose:
            print(f"Loaded     : {self.filepath}")
            print(f"Rows       : {len(df)}")
            print(f"Year range : {min(year_cols)} - {max(year_cols)}  ({len(year_cols)} columns)")

    def _process_row(self, row) -> Optional[RealWorldSample]:
        row_id    = str(row.get("ID", "")).strip()
        tech_name = str(row.get("Technology Name", "")).strip()

        def reject(reason):
            self.rejected.append(RejectedSeries(row_id, tech_name, reason))

        raw_vals = np.array([row[yr] if yr in row.index else np.nan for yr in self._year_cols], dtype=float)
        years = np.array(self._year_cols)
        valid_mask = ~np.isnan(raw_vals)
        if valid_mask.sum() < self.min_years:
            reject(f"fewer than {self.min_years} observed data points"); return None
        first_valid = int(np.argmax(valid_mask))
        last_valid  = int(len(valid_mask) - 1 - np.argmax(valid_mask[::-1]))
        raw_vals = raw_vals[first_valid: last_valid + 1]
        years    = years[first_valid: last_valid + 1]
        if len(raw_vals) < self.min_years:
            reject(f"fewer than {self.min_years} points after stripping NaN ends"); return None
        x_filled, n_gaps = interpolate_gaps(raw_vals, self.max_gap)
        if np.any(np.isnan(x_filled)):
            reject("unresolvable internal NaN gaps (gap > max_gap)"); return None
        peak_idx = int(np.argmax(x_filled)); peak_val = float(x_filled[peak_idx]); year_peak = int(years[peak_idx])
        x_pre = x_filled[: peak_idx + 1]; diffs = np.diff(x_pre)
        if len(diffs) > 0 and (diffs >= 0).mean() < 0.40:
            reject("rising phase is predominantly declining - not adoption"); return None
        if len(x_pre) < self.min_years:
            reject(f"rising phase < {self.min_years} points"); return None
        x_post = x_filled[peak_idx:]; years_post = years[peak_idx:]
        series_type = "pre_saturation"; saturation_reached = False
        end_idx = len(x_filled) - 1; n_post_peak = 0
        if len(x_post) <= 1:
            series_type = "pre_saturation"; end_idx = peak_idx; n_post_peak = 0
        else:
            if detect_structural_decline(x_post, years_post, peak_val,
                                         self.decline_threshold, self.decline_duration, DISRUPTION_WINDOWS):
                series_type = "declining_structural"; end_idx = peak_idx; n_post_peak = 0
            else:
                sat_end = detect_saturation(x_post, years_post, peak_val, self.sat_threshold, self.sat_duration)
                if sat_end is not None:
                    series_type = "saturating"; saturation_reached = True
                    end_idx = peak_idx + sat_end; n_post_peak = sat_end
                else:
                    series_type = "pre_saturation"; end_idx = len(x_filled) - 1; n_post_peak = end_idx - peak_idx
        x_final = x_filled[: end_idx + 1]; years_final = years[: end_idx + 1]
        if len(x_final) < self.min_years:
            reject(f"final segment < {self.min_years} points after Option B trimming"); return None
        x_norm = normalize_series(x_final)
        if x_norm.max() - x_norm.min() < self.min_variation:
            reject("insufficient variation after normalisation"); return None
        x_interp = pchip_interpolate_to_target(x_norm, target=TARGET_POINTS)
        return RealWorldSample(
            row_id=row_id, tech_name=tech_name,
            metric=str(row.get("Metric", "")).strip(), unit=str(row.get("Unit", "")).strip(),
            source=str(row.get("Data Source", "")).strip(), spatial_scale=str(row.get("Spatial Scale", "")).strip(),
            region=str(row.get("Region", "")).strip(), country=str(row.get("Country Name", "")).strip(),
            variable=str(row.get("Variable", "")).strip(),
            years_full=years_final, x_full=x_interp,
            year_start=int(years[0]), year_end=int(years[-1]), year_sat=int(years_final[-1]),
            series_length=TARGET_POINTS, n_observed=len(x_final), n_gaps_filled=n_gaps, peak_idx=peak_idx,
            series_type=series_type, n_post_peak=n_post_peak, year_peak=year_peak,
            saturation_reached=saturation_reached,
        )

    def run(self, verbose=True):
        if self._df is None:
            self.load(verbose=verbose)
        if verbose:
            print(f"\nProcessing {len(self._df)} rows (Option B - full S-curve curation) ...")
        for _, row in self._df.iterrows():
            s = self._process_row(row)
            if s is not None:
                self.samples.append(s)
        if verbose:
            reasons = Counter(r.reason for r in self.rejected)
            print(f"  Kept {len(self.samples)} / {len(self.samples)+len(self.rejected)}")
            for reason, count in reasons.most_common():
                print(f"    {count:5d}  {reason}")

    def save(self, output_dir):
        _save_samples(output_dir, self.samples, config={
            "filepath": self.filepath, "stage": "Stage 1 / Option B - full S-curve",
            "min_years": self.min_years, "min_variation": self.min_variation,
            "max_gap": self.max_gap, "sat_threshold": self.sat_threshold,
            "sat_duration": self.sat_duration, "decline_threshold": self.decline_threshold,
            "decline_duration": self.decline_duration, "target_points": TARGET_POINTS,
            "n_disruption_windows": len(DISRUPTION_WINDOWS),
            "interpolation": "PCHIP - pure interpolation, no extrapolation",
        })


# =============================================================================
# 4.  World-data column remap (World_Data_Curator.py)
# =============================================================================
def preprocess_world_csv(filepath: str) -> str:
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    year_cols = [c for c in df.columns if str(c).strip().isdigit() and 1500 <= int(c) <= 2100]
    out = pd.DataFrame()
    out["ID"]              = ["W" + str(i).zfill(4) for i in range(len(df))]
    out["Technology Name"] = df["Short Name"].astype(str).str.strip()
    out["Metric"]          = df["Variable"].astype(str).str.strip()
    out["Unit"]            = df["Unit"].astype(str).str.strip()
    out["Data Source"]     = df["Model"].astype(str).str.strip()
    out["Region"]          = df["Region"].astype(str).str.strip()
    out["Country Name"]    = df["Region"].astype(str).str.strip()
    out["Spatial Scale"]   = df["Region"].apply(lambda r: "Global" if str(r).strip().lower() == "world" else "National")
    out["Variable"]        = df["Variable"].astype(str).str.strip()
    for yr in year_cols:
        out[yr] = pd.to_numeric(df[yr], errors="coerce")
    tmp_path = filepath.replace(".csv", "_remapped.csv")
    out.to_csv(tmp_path, index=False)
    return tmp_path


# =============================================================================
# 5.  Merge + de-duplicate (Combined_Dataset_Analysis.py, curation-only)
# =============================================================================
DEDUP_CORR_THRESHOLD = 0.9999


def _norm(s) -> str:
    return str(s).strip().lower()


def _is_global_region(region_str) -> bool:
    return _norm(region_str) in {"world", "global"}


def _tag_scale(sample, source) -> str:
    if source == "world_csv":
        return "global"
    return "global" if _is_global_region(getattr(sample, "region", "")) else "national"


def records_from_samples(samples, source):
    return [dict(sample=s, x_full=np.asarray(s.x_full, np.float32),
                 scale=_tag_scale(s, source), source=source, orig_idx=i)
            for i, s in enumerate(samples)]


def deduplicate(records, corr_thresh=DEDUP_CORR_THRESHOLD):
    groups = defaultdict(list)
    for i, rec in enumerate(records):
        s = rec["sample"]
        groups[(_norm(s.tech_name), _norm(getattr(s, "region", "")))].append(i)
    drop = set()
    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                ia, ib = indices[a], indices[b]
                if ia in drop or ib in drop:
                    continue
                xa, xb = records[ia]["x_full"], records[ib]["x_full"]
                if np.std(xa) < 1e-8 or np.std(xb) < 1e-8:
                    corr = 1.0 if np.allclose(xa, xb, atol=1e-6) else 0.0
                else:
                    corr = float(np.corrcoef(xa, xb)[0, 1])
                if corr > corr_thresh:
                    src_a, src_b = records[ia]["source"], records[ib]["source"]
                    drop.add(ia if (src_b == "world_csv" and src_a != "world_csv") else ib)
    return [rec for i, rec in enumerate(records) if i not in drop]


def save_combined(records, out_dir):
    """Stamp scale/source, write the combined tier, and return (samples, X, meta_df)."""
    samples, X_list, meta_rows = [], [], []
    for rec in records:
        s = rec["sample"]
        s.scale = rec["scale"]; s.source = rec["source"]
        samples.append(s); X_list.append(rec["x_full"])
        meta_rows.append({
            "tech_name": s.tech_name, "region": getattr(s, "region", ""),
            "country": getattr(s, "country", ""), "unit": getattr(s, "unit", ""),
            "year_start": getattr(s, "year_start", ""), "year_peak": getattr(s, "year_peak", ""),
            "year_sat": getattr(s, "year_sat", ""), "series_type": getattr(s, "series_type", ""),
            "scale": rec["scale"], "source": rec["source"],
        })
    X = np.stack(X_list).astype(np.float32)
    meta = pd.DataFrame(meta_rows)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "real_world_samples.pkl"), "wb") as f:
        pickle.dump(samples, f, protocol=4)
    np.save(os.path.join(out_dir, "X_full.npy"), X)
    np.save(os.path.join(out_dir, "lengths.npy"), np.full(len(samples), 500, dtype=np.int32))
    meta.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)
    n_nat = int((meta["scale"] == "national").sum()); n_glo = int((meta["scale"] == "global").sum())
    print(f"  combined saved: {len(samples)} samples  (national={n_nat} global={n_glo})")
    return samples, X, meta


# =============================================================================
# 6.  Generic tier writer (used by filter10 / genuine_v1 / genuine_v2)
# =============================================================================
def _save_samples(out_dir, samples, config=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "real_world_samples.pkl"), "wb") as f:
        pickle.dump(samples, f, protocol=4)
    X = np.stack([np.asarray(s.x_full, np.float32) for s in samples]).astype(np.float32)
    np.save(os.path.join(out_dir, "X_full.npy"), X)
    np.save(os.path.join(out_dir, "lengths.npy"), np.full(len(samples), TARGET_POINTS, dtype=np.int32))
    if config is not None:
        with open(os.path.join(out_dir, "curation_config.json"), "w") as f:
            json.dump(config, f, indent=2)
    return X


def save_tier(out_dir, mask, base_samples, base_X, base_meta, info):
    """Write a subset tier (X, lengths, metadata.csv, pkl, filter_info.json)."""
    os.makedirs(out_dir, exist_ok=True)
    idx = np.where(mask)[0]
    samp = [base_samples[i] for i in idx]
    X = base_X[mask].astype(np.float32)
    meta = base_meta.iloc[idx].reset_index(drop=True)
    with open(os.path.join(out_dir, "real_world_samples.pkl"), "wb") as f:
        pickle.dump(samp, f, protocol=4)
    np.save(os.path.join(out_dir, "X_full.npy"), X)
    np.save(os.path.join(out_dir, "lengths.npy"), np.full(len(samp), TARGET_POINTS, dtype=np.int32))
    meta.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)
    info = dict(info); info.update(n_kept=int(len(samp)), original_idx=idx.tolist())
    with open(os.path.join(out_dir, "filter_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    return samp, X, meta


# =============================================================================
# 7.  Removal lists
# =============================================================================
# genuine_v1: recovered from committed data (original create_genuine_dataset.py missing).
# These 108 names are the tech_names present in `combined` but absent from `combined_genuine`
# (vaccines, raw-material / industrial production, food/agriculture, mining/refining).
GENUINE_V1_REMOVE = [
    "Acrylic Fiber", "Acrylonitrile", "Aluminium Refining", "Ammonia Synthesis", "Aniline",
    "Aquaculture Production", "BCG Vaccine", "Beer Production", "Benzene", "Cadmium Refining",
    "Cane Sugar", "Cane Sugar Production", "Caprolactam", "Capture Fisheries", "Caustic Soda",
    "Coal Production", "Cobalt", "Cobalt Mine Production", "Copper Mines", "Copper Primary Production",
    "Copper Refining", "Copper|Mining", "Copper|Refining", "Crude Oil", "Cyclohexane",
    "DTP1 Vaccine", "DTP3 Vaccine", "Ethanolamine", "Ethyl Alcohol", "Ethylene",
    "Ethylene Glycol", "Formaldehyde", "Gold", "Gold Production", "Graphite",
    "Graphite Mine Production", "HEPB3 Vaccine", "HEPBB Vaccine", "HIB3 Vaccine",
    "High-Density Polyethylene", "Hydrochloric Acid", "Hydrofluoric Acid", "Iron Ore", "Lead",
    "Lead Mines", "Lithium Mine Production", "Low-Density Polyethylene", "MCV1 Vaccine",
    "MCV2 Vaccine", "Magnesium", "Maleic Anhydride", "Methanol", "Milk Production",
    "Motor Gasoline", "Natural Gas", "Natural Gas Production", "Neoprene Rubber",
    "Nickel Production", "Nitric Acid", "Nitrogen Fertilizer", "Oil Production",
    "Oil Refining Capacity", "PCV3 Vaccine", "POL3 Vaccine", "Paraxylene", "Pentaerythritol",
    "Phenol", "Phosphate Fertilizer", "Phthalic Anhydride", "Polyester", "Polyester Fiber",
    "PolyethyleneHD", "PolyethyleneLD", "Polystyrene", "Polyvinylchloride", "Potash Fertilizer",
    "Primary Aluminum Production", "Primary Bauxite Production", "Primary Copper",
    "Primary Magnesium", "RCV1 Vaccine", "ROTAC Vaccine", "Rare Earth Mine Production",
    "Raw Steel", "Raw Steel Production", "Salt Production", "Sand and Gravel Construction",
    "Sand and Gravel Industrial", "Sand and Gravel|Construction", "Sand and Gravel|Industrial",
    "Shale Oil", "Shale Production", "Silver", "Silver Production", "Sodium", "Sodium Chlorate",
    "Styrene", "Sulphuric Acid", "Synthetic Filaments", "Tin", "Tin Refining", "Titanium Sponge",
    "Urea", "Vinyl Acetate", "Vinyl Chloride", "YFV Vaccine", "Zinc", "Zinc Refining",
]

# genuine_v2: second-layer removal (create_genuine_v2_dataset.py) — 7 groups.
GENUINE_V2_GROUP_REMOVE = {
    "moores_law_metrics": ["Processor Performance", "Microprocessor Clock Speed",
                           "Random Access Memory", "Transistors Per Microprocessor",
                           "Transistors per Microprocessor"],
    "activity_traffic_volumes": ["Postal Traffic", "Postal and Telegraph Traffic",
                                 "Telegraph Traffic", "Internet Data Traffic", "Internet Traffic",
                                 "Internet Backbone Bandwidth"],
    "military_stockpile": ["Nuclear Weapons"],
    "industrial_production_slipped_through": ["BisphenolA", "Cement", "Cement Production",
                                              "Coal Power", "Natural Gas Power", "Ethanol Production"],
    "fossil_fuel_infrastructure": ["Oil Pipeline", "Oil Pipelines", "Oil Refineries",
                                   "Natural Gas Pipeline", "Natural Gas Pipelines",
                                   "Liquefied Natural Gas", "Liquefied Natural Gas Exports",
                                   "Liquefied Natural Gas Plants"],
    "bioenergy_production_volumes": ["All Biofuels", "Biogas", "Solid Biofuels", "Solid Biomass",
                                     "Liquid Biofuels"],
    "launch_activity_counts": ["Satellite Launches", "Space Launches"],
}
GENUINE_V2_REMOVE = sorted({t for ts in GENUINE_V2_GROUP_REMOVE.values() for t in ts})


# =============================================================================
# 8.  Shape diagnostic / s-score (plot_scurve_quality.py, data only)
# =============================================================================
def hard_exclude(xn):
    end_frac = float(xn[-1]); start_frac = float(xn[0]); peak_pos = float(np.argmax(xn)) / len(xn)
    if start_frac > 0.35:
        return True, "Truncated-start"
    if end_frac < 0.70 and peak_pos < 0.80:
        return True, "Bell/Declining"
    return False, None


def _logistic(t, k, t0):
    return 1.0 / (1.0 + np.exp(-k * (t - t0)))


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def scurve_score(years, x):
    years = np.array(years, float); x = np.array(x, float)
    if x.max() <= 0:
        return dict(s_score=np.nan, r2_logistic=np.nan, r2_linear=np.nan,
                    has_inflection=False, growth_range=np.nan)
    xn = x / x.max(); n = len(xn); t_rel = years - years[0]
    r2_lin = _r2(xn, np.polyval(np.polyfit(t_rel, xn, 1), t_rel))
    r2_log = np.nan
    try:
        mid = t_rel[np.argmin(np.abs(xn - 0.5))] if (xn >= 0.5).any() else t_rel[n // 2]
        popt, _ = curve_fit(_logistic, t_rel, xn, p0=[0.1, mid], maxfev=6000,
                            bounds=([1e-5, t_rel[0] - 1], [5.0, t_rel[-1] + 1]))
        r2_log = _r2(xn, _logistic(t_rel, *popt))
    except Exception:
        pass
    d2 = np.diff(np.diff(xn))
    has_inflection = bool(d2.max() > 0.003 and d2.min() < -0.003)
    growth_range = float(((xn > 0.10) & (xn < 0.90)).mean())
    s_score = (r2_log - r2_lin) if not np.isnan(r2_log) else -1.0
    return dict(s_score=round(s_score, 4),
                r2_logistic=round(r2_log, 4) if not np.isnan(r2_log) else np.nan,
                r2_linear=round(r2_lin, 4), has_inflection=has_inflection,
                growth_range=round(growth_range, 3))


def compute_shape_diagnostic(gv2_samples, out_dir):
    records = []
    for i, s in enumerate(gv2_samples):
        years = np.array(s.years_full, dtype=int); n = len(years)
        x = np.array(s.x_full, dtype=float)[:n]
        if x.max() <= 0 or n < 8:
            continue
        xn = x / x.max()
        excluded, reason = hard_exclude(xn)
        if excluded:
            rec = dict(idx=i, tech=s.tech_name, country=s.country, scale=s.spatial_scale,
                       series_type=s.series_type, n_pts=n, year_start=s.year_start,
                       excluded=True, excl_reason=reason, end_frac=round(float(xn[-1]), 3),
                       start_frac=round(float(xn[0]), 3), s_score=np.nan, r2_logistic=np.nan,
                       r2_linear=np.nan, has_inflection=False, growth_range=np.nan)
        else:
            rec = dict(idx=i, tech=s.tech_name, country=s.country, scale=s.spatial_scale,
                       series_type=s.series_type, n_pts=n, year_start=s.year_start,
                       excluded=False, excl_reason=None, end_frac=round(float(xn[-1]), 3),
                       start_frac=round(float(xn[0]), 3), **scurve_score(years, x))
        records.append(rec)
    df = pd.DataFrame(records)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "scurve_quality.csv"), index=False)
    ne = ~df["excluded"]
    hardexcl = int(ne.sum())
    s002 = int((ne & (df["s_score"] >= 0.02)).sum())
    s005 = int((ne & (df["s_score"] >= 0.05)).sum())
    return len(df), hardexcl, s002, s005


# =============================================================================
# 9.  Orchestration
# =============================================================================
def main():
    print("=" * 70)
    print("  Data curation from raw  (SI Section S1)")
    print("=" * 70)

    # --- Stage 1: regional curation (technologies.csv -> real_world, 4124) ----
    print("\n[1/6] Regional curation (technologies.csv) -> data/curated/real_world/")
    reg = RealWorldCurator(TECH_REGIONAL)
    reg.run(); reg.save(os.path.join(CURATED, "real_world"))

    # --- Stage 2: global curation (tech_world.csv -> world_data, 137) ---------
    print("\n[2/6] Global curation (tech_world.csv) -> data/curated/world_data/")
    tmp = preprocess_world_csv(TECH_WORLD)
    world = RealWorldCurator(tmp, metadata_cols=METADATA_COLS)
    world.run(); world.save(os.path.join(CURATED, "world_data"))
    try:
        os.remove(tmp)
    except OSError:
        pass

    # --- Stage 3: merge + dedup -> combined (4217) ----------------------------
    print("\n[3/6] Merge + de-duplicate -> data/curated/combined/")
    records = records_from_samples(reg.samples, "regional_csv") + \
        records_from_samples(world.samples, "world_csv")
    print(f"  before dedup: {len(records)}")
    records = deduplicate(records)
    print(f"  after dedup : {len(records)}")
    comb_samples, comb_X, comb_meta = save_combined(records, os.path.join(CURATED, "combined"))

    # --- Stage 4: filter10 (start < 10%) -> 3540 ------------------------------
    print("\n[4/6] Left-truncation filter -> data/curated/combined_filter10/")
    mask_f10 = comb_X[:, 0] < 0.10
    save_tier(os.path.join(CURATED, "combined_filter10"), mask_f10, comb_samples, comb_X, comb_meta,
              info={"filter_name": "filter10", "threshold": 0.10, "n_original": len(comb_samples)})
    print(f"  filter10: {int(mask_f10.sum())}")

    # --- Stage 5: genuine_v1 (drop non-adoption categories) -> 1658 -----------
    print("\n[5/6] Genuine-adoption filter pass 1 -> data/curated/combined_genuine/")
    mask_g1 = ~comb_meta["tech_name"].isin(set(GENUINE_V1_REMOVE))
    g1_samples, g1_X, g1_meta = save_tier(
        os.path.join(CURATED, "combined_genuine"), mask_g1.values, comb_samples, comb_X, comb_meta,
        info={"filter_name": "genuine_v1", "n_original": len(comb_samples),
              "n_removed_tech_names": len(GENUINE_V1_REMOVE),
              "note": "removal list recovered from committed data (original script unavailable)"})
    print(f"  genuine_v1: {int(mask_g1.sum())}")

    # --- Stage 6: genuine_v2 (drop 7 further categories) -> 1372 (MAIN) -------
    print("\n[6/6] Genuine-adoption filter pass 2 -> data/curated/combined_genuine_v2/")
    mask_g2 = ~g1_meta["tech_name"].isin(set(GENUINE_V2_REMOVE))
    g2_samples, g2_X, g2_meta = save_tier(
        os.path.join(CURATED, "combined_genuine_v2"), mask_g2.values, g1_samples, g1_X, g1_meta,
        info={"filter_name": "genuine_v2", "n_original": len(g1_samples),
              "n_removed_tech_names": len(GENUINE_V2_REMOVE)})
    print(f"  genuine_v2: {int(mask_g2.sum())}  ({g2_meta['tech_name'].nunique()} technologies)")

    # --- Shape diagnostic (s-score) -> hardexcl / s002 / s005 -----------------
    print("\n[+] Shape diagnostic (s-score) -> results/unsup/shape_diagnostic/scurve_quality.csv")
    n_scored, hardexcl, s002, s005 = compute_shape_diagnostic(g2_samples, SHAPEOUT)
    print(f"  scored={n_scored}  hardexcl={hardexcl}  s002={s002}  s005={s005}")

    # --- Summary vs SI S1 -----------------------------------------------------
    print("\n" + "=" * 70)
    print("  Tier counts  (SI S1 target in parentheses)")
    print("=" * 70)
    for label, got, want in [
        ("real_world", len(reg.samples), 4124),
        ("combined", len(comb_samples), 4217),
        ("filter10", int(mask_f10.sum()), 3540),
        ("genuine_v1", int(mask_g1.sum()), 1658),
        ("genuine_v2", int(mask_g2.sum()), 1372),
        ("hardexcl", hardexcl, 815),
        ("s002", s002, 327),
        ("s005", s005, 224),
    ]:
        print(f"  {label:14s} {got:6d}   (SI {want})   {'OK' if got == want else 'DIFF'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
