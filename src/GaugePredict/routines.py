# -*- coding: utf-8 -*-
"""
routines.py

Core utilities for GaugePredict functions downloader.py, predict.py, and plotting.py.
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
    """
    Resolve a project root directory by walking up parent folders.

    **Inputs** :

        file_path : 'str or pathlib.Path'
            Path within the project (often __file__ from a module).

        levels_up : 'int'
            Number of parent levels to traverse. levels_up=1 returns the parent
            directory of file_path.

    **Outputs** :

        project_root : 'pathlib.Path'
            Resolved project root directory.
    """
    p = Path(file_path).resolve()
    return p.parents[int(levels_up)]


def resolve_under_project(project_root, rel_path):
    """
    Resolve a relative path under a known project root.

    **Inputs** :

        project_root : 'str or pathlib.Path'
            Project root directory.

        rel_path : 'str or pathlib.Path'
            Relative path under the project root.

    **Outputs** :

        path : 'pathlib.Path'
            Absolute, resolved path under the project root.
    """
    return (Path(project_root) / Path(rel_path)).resolve()


def load_run_config(run_root):
    """
    Load a run configuration JSON from a run directory.

    Expects a file named "compute_summary.json" inside run_root.

    **Inputs** :

        run_root : 'str or pathlib.Path'
            Directory containing compute_summary.json.

    **Outputs** :

        config : 'dict'
            Parsed JSON configuration dictionary.

    **Raises** :

        FileNotFoundError
            If compute_summary.json does not exist under run_root.
    """
    run_root = Path(run_root)
    summary_path = run_root / "compute_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"compute_summary.json not found: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Geospatial Boundaries
# =============================================================================
def load_hucs_3857(base_dir):
    """
    Load HUC2 watershed boundary shapefiles and project to EPSG:3857.

    Expected layout under base_dir:
        HUC??/WBDHU2.shp

    **Inputs** :

        base_dir : 'str or pathlib.Path'
            Root directory containing HUC?? folders.

    **Outputs** :

        gdf : 'geopandas.GeoDataFrame'
            Concatenated watershed boundaries in EPSG:3857.

    **Raises** :

        FileNotFoundError
            If no matching shapefiles are found, or none could be loaded successfully.
    """
    base_dir = Path(base_dir)
    hits = list(base_dir.glob("HUC??/WBDHU2.shp"))
    if not hits:
        raise FileNotFoundError(f"No HUC shapefiles found under {base_dir}")

    codes = sorted({p.parent.name[-2:] for p in hits})
    gdfs = []

    for code in codes:
        shp = base_dir / f"HUC{code}" / "WBDHU2.shp"
        if not shp.exists():
            warnings.warn(f"Missing {shp}, skipping")
            continue

        try:
            df = gpd.read_file(shp).rename(columns=str.lower)
        except Exception as e:
            warnings.warn(f"Failed to read {shp}: {e}")
            continue

        if "huc2" not in df.columns:
            if "huc02" in df.columns:
                df = df.rename(columns={"huc02": "huc2"})
            else:
                df["huc2"] = code

        df["huc2"] = df["huc2"].astype(str).str.zfill(2)
        gdfs.append(df)

    if not gdfs:
        raise FileNotFoundError("No valid HUC shapefiles after scanning")

    out = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    if out.crs is None:
        warnings.warn("HUC GeoDataFrame has no CRS; assuming EPSG:4326 before projecting to 3857")
        out = out.set_crs(4326)

    if out.crs.to_epsg() == 3857:
        return out
    return out.to_crs(3857)


# =============================================================================
# Time index utilities
# =============================================================================
def generate_full_index(start_date, end_date, *, localize=True, tz="UTC"):
    """
    Generate a daily date range.

    If localize=True, returns a timezone-aware DatetimeIndex in timezone `tz`.
    If localize=False, returns a naive DatetimeIndex.

    **Inputs** :

        start_date, end_date : 'str or datetime-like'
            Inclusive bounds accepted by pandas.date_range().

        localize : 'bool'
            If True, localize/convert the index to timezone `tz`.

        tz : 'str'
            Timezone name used for localization or conversion.

    **Outputs** :

        idx : 'pandas.DatetimeIndex'
            Daily index from start_date through end_date.
    """
    idx = pd.date_range(start=start_date, end=end_date, freq="D")
    if localize:
        if idx.tz is None:
            idx = idx.tz_localize(tz)
        else:
            idx = idx.tz_convert(tz)
    return idx


def build_target_dates(full_index, sequence_length, forecast_horizon, n_samples):
    """
    Build the target date vector aligned to generated supervised sequences.

    For a sequence length T and forecast horizon H, the first target corresponds
    to full_index[T + H - 1].

    **Inputs** :

        full_index : 'pandas.DatetimeIndex'
            Full daily index covering the modeling period.

        sequence_length : 'int'
            Number of past days used per input sample (T).

        forecast_horizon : 'int'
            Lead time in days between last input day and target day (H).

        n_samples : 'int'
            Number of supervised samples (typically len(y_seq)).

    **Outputs** :

        dates : 'numpy.ndarray'
            Array of datetime-like values corresponding to each target.
    """
    start = int(sequence_length) + int(forecast_horizon) - 1
    return np.asarray(full_index[start : start + int(n_samples)])


def generate_train_test_masks(full_index, sequence_length, y_seq, forecast_horizon, cutoff_date, tz="UTC"):
    """
    Generate boolean masks for train/test split based on a cutoff target date.

    Split is on target dates:
    - train if target_date < cutoff
    - test otherwise

    **Inputs** :

        full_index : 'pandas.DatetimeIndex'
            Full daily index.

        sequence_length : 'int'
            Input sequence length (T).

        y_seq : 'array-like'
            Target sequence array used only for its length.

        forecast_horizon : 'int'
            Forecast horizon (H).

        cutoff_date : 'str or datetime-like'
            Cutoff date for splitting.

        tz : 'str'
            Timezone assumption if full_index and cutoff_date are naive.

    **Outputs** :

        train_mask : 'numpy.ndarray (bool)'
            Boolean array of length len(y_seq), True for training samples.

        test_mask : 'numpy.ndarray (bool)'
            Boolean array of length len(y_seq), True for test samples.
    """
    T = int(sequence_length)
    H = int(forecast_horizon)
    start = T + H - 1

    full_utc = _as_utc_daily_index(full_index, tz=tz)
    target_dates = np.asarray(full_utc[start : start + int(len(y_seq))])

    cutoff = pd.Timestamp(cutoff_date)
    if cutoff.tz is None:
        cutoff = cutoff.tz_localize(tz)
    cutoff = cutoff.tz_convert("UTC")

    train_mask = target_dates < cutoff
    test_mask = target_dates >= cutoff
    return train_mask, test_mask


# =============================================================================
# Timezone helpers (kept for backward compatibility)
# =============================================================================
def _to_utc_index(idx, *, tz="UTC"):
    """
    Convert an index-like object to a tz-aware UTC DatetimeIndex.

    If timestamps are naive, localize to `tz` first, then convert to UTC.
    """
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize(tz)
    return idx.tz_convert("UTC")


def _full_index_utc(full_index, *, tz="UTC"):
    """
    Convert a full_index to a tz-aware UTC DatetimeIndex.
    """
    return _to_utc_index(pd.to_datetime(full_index), tz=tz)


def _as_utc_daily_index(full_index, tz="UTC"):
    """
    Convert a daily full_index to a tz-aware UTC DatetimeIndex.
    """
    idx = pd.DatetimeIndex(full_index)
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    return idx.tz_convert("UTC")


def _normalize_site_no(site_no):
    """
    Return both raw and normalized site identifiers.
    """
    site_no_raw = str(site_no).strip()
    site_no_norm = site_no_raw.lstrip("0")
    return site_no_raw, site_no_norm


def _normalize_site_id(site_no):
    """
    Backward compatible alias: normalize a site identifier to a clean string.
    """
    return str(site_no).strip()


def _normalize_site_id_norm(site_no):
    """
    Backward compatible alias: normalize a site identifier with leading zeros removed.
    """
    return str(site_no).strip().lstrip("0")


def _series_from_parameter_dict(param_dict, *, tz="UTC"):
    """
    Convert a date-keyed dictionary to a daily UTC pandas.Series.

    Input format:
        {"YYYY-MM-DD": value, ...}
    """
    if not isinstance(param_dict, dict) or len(param_dict) == 0:
        return pd.Series(dtype=float)

    dt = pd.to_datetime(list(param_dict.keys()), errors="coerce")
    vals = list(param_dict.values())
    s = pd.Series(vals, index=dt)
    s = s[~s.index.isna()]
    if s.empty:
        return pd.Series(dtype=float)

    s.index = _to_utc_index(s.index, tz=tz)
    s = pd.to_numeric(s, errors="coerce")
    return s.sort_index()


# =============================================================================
# Predictor Loading
# =============================================================================
def load_data(
    data_files,
    full_index,
    *,
    allow_site_ids_norm=None,
    tz="UTC",
    fill=True,
):
    """
    Load predictor time series from one or more cached JSON files.

    Expected JSON layout (Layout B only):
        {site_no: payload, ...}

    Each payload must contain a date-keyed mapping under `data_key`, typically "parameter":
        payload[data_key] = {"YYYY-MM-DD": value, ...}
    """
    if not isinstance(data_files, (list, tuple)) or len(data_files) == 0:
        raise ValueError("data_files must be a non-empty list of dicts with keys {'path', optional 'data_key'}")

    allowed = None
    if allow_site_ids_norm is not None:
        allowed = {str(s).lstrip("0") for s in allow_site_ids_norm}

    full_utc = _as_utc_daily_index(full_index, tz=tz)

    all_series = []
    all_meta = []

    def _coerce_float(x):
        try:
            return float(pd.to_numeric(x, errors="coerce"))
        except Exception:
            return np.nan

    def _try_add_site(site_no, payload, *, data_key, source_path):
        if not isinstance(payload, dict):
            return

        site_no_raw, site_no_norm = _normalize_site_no(site_no)
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

        s = pd.to_numeric(s, errors="coerce")
        s = s.replace([-99999, -999999, -9999, -999], np.nan)
        s = s.mask(s <= -1e4, np.nan)

        if s.index.tz is None:
            s.index = s.index.tz_localize(tz)
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
                "lat": _coerce_float(lat),
                "lon": _coerce_float(lon),
                "huc": payload.get("huc", None),
                "data_key": str(data_key),
                "source_path": str(source_path),
            }
        )

    for spec in data_files:
        if not isinstance(spec, dict) or "path" not in spec:
            raise ValueError("Each entry in data_files must be a dict containing at least {'path': ...}")

        json_path = Path(spec["path"])
        data_key = str(spec.get("data_key", "parameter"))

        if not json_path.exists():
            raise FileNotFoundError(f"Predictor JSON not found: {json_path}")

        with json_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        if not isinstance(obj, dict) or len(obj) == 0:
            continue

        for site_no, payload in obj.items():
            _try_add_site(site_no, payload, data_key=data_key, source_path=json_path)

    if not all_series:
        raise RuntimeError(
            "No predictor series loaded.\n"
            "Common causes:\n"
            "  - data_key is wrong (your JSON does not contain that key)\n"
            "  - allowed_sites filtered everything\n"
            "  - JSON is not layout B: expected {site_no: payload}\n"
        )

    X = np.asarray(all_series, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != len(full_utc):
        raise RuntimeError(f"Predictor array has wrong shape {X.shape}; expected [C, {len(full_utc)}]")

    return X, all_meta


# =============================================================================
# Target CSV loading
# =============================================================================
def load_target_csv(csv_path, full_index, *, date_col="date", value_col="value", tz="UTC", fill=True):
    """
    Load a daily target time series from a CSV file and align to full_index.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Target CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"Target CSV missing date_col='{date_col}'")
    if value_col not in df.columns:
        raise ValueError(f"Target CSV missing value_col='{value_col}'")

    d = pd.to_datetime(df[date_col], errors="coerce")
    valid = ~d.isna()
    if not bool(valid.any()):
        raise ValueError("Target CSV has no parseable dates")

    df = df.loc[valid].copy()
    d = pd.to_datetime(df[date_col], errors="coerce")

    if getattr(d.dt, "tz", None) is None:
        d = d.dt.tz_localize(tz)
    d = d.dt.tz_convert("UTC")

    v = pd.to_numeric(df[value_col], errors="coerce").to_numpy()
    s = pd.Series(v, index=d).sort_index()

    full_utc = _as_utc_daily_index(full_index, tz=tz)
    s = s.reindex(full_utc)

    if fill:
        s = s.interpolate(limit_direction="both").ffill().bfill()

    s.name = str(value_col)
    return s


# =============================================================================
# Predictor stack processing + sequence construction
# =============================================================================
def process_data(raw_X_data, target_series, *, smooth_window_days=1):
    """
    Construct a final predictor stack by appending target-derived channels.

    Appends:
    - smoothed target (rolling mean)
    - first difference of smoothed target
    """
    y = pd.to_numeric(pd.Series(target_series), errors="coerce").to_numpy(dtype=float)
    if y.ndim != 1:
        y = y.reshape(-1)

    w = int(max(1, smooth_window_days))
    y_sm = pd.Series(y).rolling(window=w, center=True, min_periods=1).mean().to_numpy(dtype=float)
    y_diff = np.diff(y_sm, prepend=y_sm[0]).astype(float)

    X = np.asarray(raw_X_data, dtype=np.float32)
    y_sm = np.asarray(y_sm, dtype=np.float32).reshape(1, -1)
    y_diff = np.asarray(y_diff, dtype=np.float32).reshape(1, -1)

    if X.shape[1] != y_sm.shape[1]:
        raise ValueError("Predictor length does not match target length")

    return np.vstack([X, y_sm, y_diff]).astype(np.float32)


def generate_sequences(sequence_length, forecast_horizon, x_raw, y):
    """
    Convert continuous predictor/target arrays into supervised learning sequences.

    **Outputs** :
        X_seq: [N, T, C]
        y_seq: [N, 1]
    """
    T = int(sequence_length)
    H = int(forecast_horizon)

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    x_raw = np.asarray(x_raw, dtype=np.float32)

    n = len(y) - T - H + 1
    if n <= 0:
        raise ValueError("Not enough data to build sequences for the requested sequence_length and forecast_horizon")

    x_seq = []
    y_seq = []
    for i in range(n):
        x_seq.append(x_raw[:, i : i + T].T)
        y_seq.append(y[i + T + H - 1])

    return (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32).reshape(-1, 1),
    )
    
