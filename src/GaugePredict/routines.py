# -*- coding: utf-8 -*-
"""
routines.py
"""

from __future__ import division, print_function, absolute_import

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

# =============================================================================
# Project paths and run configuration
# =============================================================================

def get_project_root(file_path, levels_up=1):
    p = Path(file_path).resolve()
    return p.parents[int(levels_up)]


def resolve_under_project(project_root, rel_path):
    return (Path(project_root) / Path(rel_path)).resolve()


def load_run_config(run_root):
    run_root = Path(run_root)
    summary_path = run_root / "compute_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"compute_summary.json not found: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# HUC discovery and loading
# =============================================================================

def load_hucs_3857(base_dir):
    hits = list(Path(base_dir).glob("HUC??/WBDHU2.shp"))
    if not hits:
        raise FileNotFoundError(f"No HUC shapefiles found under {base_dir}")

    codes = sorted({p.parent.name[-2:] for p in hits})
    gdfs = []

    for code in codes:
        shp = Path(base_dir) / f"HUC{code}" / "WBDHU2.shp"
        if not shp.exists():
            warnings.warn(f"Missing {shp}, skipping")
            continue

        df = gpd.read_file(shp).rename(columns=str.lower)

        if "huc2" not in df.columns:
            if "huc02" in df.columns:
                df = df.rename(columns={"huc02": "huc2"})
            else:
                df["huc2"] = code

        df["huc2"] = df["huc2"].astype(str).str.zfill(2)
        gdfs.append(df)

    if not gdfs:
        raise FileNotFoundError("No valid HUC shapefiles after scanning")

    out = pd.concat(gdfs, ignore_index=True)
    return out if (out.crs and out.crs.to_epsg() == 3857) else out.to_crs(3857)


# =============================================================================
# Time index utilities
# =============================================================================

def generate_full_index(start_date, end_date, *, localize=True, tz="UTC"):
    idx = pd.date_range(start=start_date, end=end_date, freq="D")
    if localize:
        if idx.tz is None:
            idx = idx.tz_localize(tz)
        else:
            idx = idx.tz_convert(tz)
    return idx


def build_target_dates(full_index, sequence_length, forecast_horizon, n_samples):
    start = int(sequence_length) + int(forecast_horizon) - 1
    return np.asarray(full_index[start : start + int(n_samples)])


def generate_train_test_masks(full_index, sequence_length, y_seq, forecast_horizon, cutoff_date):
    target_dates = full_index[int(sequence_length) : len(full_index) - int(forecast_horizon) + 1]
    target_dates = np.asarray(target_dates[: len(y_seq)])
    cutoff = pd.Timestamp(cutoff_date)
    if cutoff.tz is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    train_mask = target_dates < cutoff
    test_mask = target_dates >= cutoff
    return train_mask, test_mask


# =============================================================================
# Loading cached predictor JSON (GaugebyHUC output)
# =============================================================================

def _to_utc_index(idx, *, tz="UTC"):
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize(tz)
    return idx.tz_convert("UTC")


def _full_index_utc(full_index, *, tz="UTC"):
    idx = pd.to_datetime(full_index)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize(tz)
    return idx.tz_convert("UTC")


def _series_from_parameter_dict(param_dict, *, tz="UTC"):
    """
    param_dict: {"YYYY-MM-DD": value, ...}
    returns: Series indexed by tz-aware UTC daily timestamps
    """
    if not isinstance(param_dict, dict) or len(param_dict) == 0:
        return pd.Series(dtype=float)

    # keys are dates, treat as midnight
    dt = pd.to_datetime(list(param_dict.keys()))
    s = pd.Series(list(param_dict.values()), index=dt)
    s.index = _to_utc_index(s.index, tz=tz)
    s = pd.to_numeric(s, errors="coerce")
    return s.sort_index()


def _normalize_site_id(site_no):
    return str(site_no).strip()


def _normalize_site_id_norm(site_no):
    return str(site_no).strip().lstrip("0")





def _as_utc_daily_index(full_index, tz="UTC"):
    idx = pd.DatetimeIndex(full_index)
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)
    return idx.tz_convert("UTC")


def _normalize_site_no(site_no):
    site_no_raw = str(site_no)
    site_no_norm = site_no_raw.lstrip("0")
    return site_no_raw, site_no_norm


def load_data(
    data_files,
    full_index,
    *,
    allow_site_ids_norm=None,
    tz="UTC",
    fill=True,
):
    """
    Load predictor time series from cached JSON(s).

    Accepts both JSON layouts:
      A) {huc: {site_no: payload}}
      B) {site_no: payload}

    payload must contain:
      - latitude/longitude (optional but used for meta)
      - data_key (e.g., "parameter"): { "YYYY-MM-DD": value, ... }
    """
    if not isinstance(data_files, (list, tuple)) or len(data_files) == 0:
        raise ValueError("data_files must be a non-empty list of dicts with keys {'path','data_key'}")

    allowed = None
    if allow_site_ids_norm is not None:
        allowed = {str(s).lstrip("0") for s in allow_site_ids_norm}

    full_utc = _as_utc_daily_index(full_index, tz=tz)

    all_series = []
    all_meta = []

    def _try_add_site(site_no, payload, *, huc=None, data_key=None, source_path=None):
        if not isinstance(payload, dict):
            return

        site_no_raw, site_no_norm = _normalize_site_no(site_no)

        # filter using normalized IDs
        if allowed is not None and site_no_norm not in allowed:
            return

        data = payload.get(data_key, None)
        if data is None or not isinstance(data, dict) or len(data) == 0:
            return

        lat = payload.get("latitude", payload.get("lat", np.nan))
        lon = payload.get("longitude", payload.get("lon", np.nan))

        s = pd.Series(data, dtype="float64")
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[~s.index.isna()]
        if s.empty:
            return

        # treat date-only as UTC midnights
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        else:
            s.index = s.index.tz_convert("UTC")

        s = s.sort_index().reindex(full_utc)

        if fill:
            s = s.interpolate(limit_direction="both").ffill().bfill()

        arr = s.to_numpy(dtype=float)
        if not np.isfinite(arr).any():
            return

        all_series.append(arr.astype(np.float32))

        all_meta.append(
            {
                "site_no": site_no_raw,
                "site_no_norm": site_no_norm,
                "lat": float(lat) if lat is not None else np.nan,
                "lon": float(lon) if lon is not None else np.nan,
                "huc": str(huc) if huc is not None else None,
                "data_key": str(data_key),
                "source_path": str(source_path) if source_path is not None else None,
            }
        )

    for spec in data_files:
        json_path = Path(spec["path"])
        data_key = str(spec.get("data_key", "parameter"))

        if not json_path.exists():
            raise FileNotFoundError(f"Predictor JSON not found: {json_path}")

        with json_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        if not isinstance(obj, dict) or len(obj) == 0:
            continue

        # Always attempt BOTH interpretations:
        # 1) top-level is sites (layout B)
        for site_no, payload in obj.items():
            _try_add_site(
                site_no,
                payload,
                huc=None,
                data_key=data_key,
                source_path=json_path,
            )

        # 2) top-level is HUCs (layout A)
        for huc, maybe_sites in obj.items():
            if not isinstance(maybe_sites, dict):
                continue
            for site_no, payload in maybe_sites.items():
                _try_add_site(
                    site_no,
                    payload,
                    huc=huc,
                    data_key=data_key,
                    source_path=json_path,
                )

    if not all_series:
        raise RuntimeError(
            "No predictor series loaded.\n"
            "This usually means one of these:\n"
            "  - data_key is wrong (your JSON does not contain that key)\n"
            "  - allowed_sites filtered everything\n"
            "  - JSON has no date->value dicts in the expected place\n"
        )

    X = np.asarray(all_series, dtype=np.float32)  # [C, T]
    if X.ndim != 2 or X.shape[1] != len(full_utc):
        raise RuntimeError(f"Predictor array has wrong shape {X.shape}; expected [C, {len(full_utc)}]")

    return X, all_meta



# =============================================================================
# Target CSV loading (for non-USGS targets)
# =============================================================================

def load_target_csv(csv_path, full_index, *, date_col="date", value_col="value", tz="UTC", fill=True):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Target CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"Target CSV missing date_col='{date_col}'")
    if value_col not in df.columns:
        raise ValueError(f"Target CSV missing value_col='{value_col}'")

    d = pd.to_datetime(df[date_col])
    if getattr(d.dt, "tz", None) is None:
        d = d.dt.tz_localize(tz)
    d = d.dt.tz_convert("UTC")

    s = pd.Series(pd.to_numeric(df[value_col], errors="coerce").to_numpy(), index=d)
    s = s.sort_index()

    full_utc = _full_index_utc(full_index, tz=tz)
    s = s.reindex(full_utc)
    if fill:
        s = s.interpolate(limit_direction="both").ffill().bfill()
    s.name = str(value_col)
    return s


# =============================================================================
# Predictor stack processing + sequences
# =============================================================================

def process_data(raw_X_data, target_series, *, smooth_window_days=7):
    """
    Build model-ready channels by appending 2 target-derived channels:
      - target_smoothed
      - target_diff

    raw_X_data: [C, T]
    target_series: Series aligned to full_index (UTC)
    returns: processed_X_data [C+2, T]
    """
    y = pd.to_numeric(pd.Series(target_series), errors="coerce").to_numpy(dtype=float)
    if y.ndim != 1:
        y = y.reshape(-1)

    # simple daily rolling mean for a "storage" proxy
    w = int(max(1, smooth_window_days))
    y_sm = pd.Series(y).rolling(window=w, center=True, min_periods=1).mean().to_numpy(dtype=float)

    # first difference (daily)
    y_diff = np.diff(y_sm, prepend=y_sm[0]).astype(float)

    X = np.asarray(raw_X_data, dtype=np.float32)
    y_sm = np.asarray(y_sm, dtype=np.float32).reshape(1, -1)
    y_diff = np.asarray(y_diff, dtype=np.float32).reshape(1, -1)

    if X.shape[1] != y_sm.shape[1]:
        raise ValueError("Predictor length does not match target length")

    return np.vstack([X, y_sm, y_diff]).astype(np.float32)


def generate_sequences(sequence_length, forecast_horizon, x_raw, y):
    x_seq, y_seq = [], []
    T = int(sequence_length)
    H = int(forecast_horizon)

    y = np.asarray(y, dtype=np.float32).reshape(-1)

    for i in range(len(y) - T - H + 1):
        # x_raw: [C, time] -> [T, C]
        x_seq.append(x_raw[:, i : i + T].T)
        y_seq.append(y[i + T + H - 1])

    return (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32).reshape(-1, 1),
    )
