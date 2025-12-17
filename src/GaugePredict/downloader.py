# -*- coding: utf-8 -*-
"""
downloader.py
"""

from __future__ import division, print_function, absolute_import

import json
from pathlib import Path

import numpy as np
import pandas as pd
from dataretrieval import nwis


# =============================================================================
# Column selection
# =============================================================================

def _rank_parameter_col(name, preference):
    name_l = str(name).lower()
    for i, tag in enumerate(preference):
        if tag in name_l:
            return i
    return len(preference)


def _pick_parameter_col(df, parameter_code, *, parameter_kind=None):
    """
    Pick the preferred daily-values column for a given parameter code.

    parameter_kind: optional hint ("discharge", "water_level", "precipitation")
    used to tweak preference ordering (sum/total for precip, mean/value otherwise).
    """
    matches = [c for c in df.columns if str(parameter_code) in str(c)]
    if not matches:
        return None

    if str(parameter_kind).lower() in {"precip", "precipitation"}:
        preference = ["sum", "total", "accum", "mean", "value", str(parameter_code)]
    else:
        preference = ["mean", "value", "sum", "total", str(parameter_code)]

    return sorted(matches, key=lambda c: _rank_parameter_col(c, preference))[0]


# =============================================================================
# Time + units
# =============================================================================

def _ensure_datetime_index(df):
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    if "datetime" in df.columns:
        df = df.copy()
        df.index = pd.to_datetime(df["datetime"])
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df


def _to_utc_index(idx, *, tz="UTC"):
    """
    Return a tz-aware UTC DatetimeIndex.
    """
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize(tz)
    return idx.tz_convert("UTC")


def _as_utc_daily_index(full_index, *, tz="UTC"):
    """
    Convert a provided full_index to tz-aware UTC without changing day stamps.
    """
    idx = pd.to_datetime(full_index)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize(tz)
    return idx.tz_convert("UTC")


def _convert_units_from_parameter_code(ts, parameter_code, *, to_units="metric"):
    """
    Convert common USGS DV units to metric.

    Expected native DV units:
      - 00060 discharge: cfs -> m^3/s
      - 00065 gage height / stage: ft -> m
      - 00045 precipitation: inches -> mm
    """
    code = str(parameter_code)

    if to_units is None or str(to_units).lower() in {"native", "none"}:
        return ts

    to_units_l = str(to_units).lower()
    if to_units_l != "metric":
        raise ValueError(f"Unsupported to_units='{to_units}'. Use 'metric' or 'native'.")

    ts = ts.astype(float)

    if code == "00060":
        return ts * 0.0283168466
    if code == "00065":
        return ts * 0.3048
    if code == "00045":
        return ts * 25.4

    raise ValueError(
        f"No conversion defined for parameter_code={parameter_code} with to_units='{to_units}'."
    )


# =============================================================================
# Public API
# =============================================================================

def load_target(
    target_site,
    full_index,
    start_date,
    end_date,
    parameter_code,
    *,
    to_units="metric",
    tz="UTC",
    parameter_kind=None,
):
    """
    Fetch NWIS DV for (site, parameter_code), align to full_index (daily),
    fill gaps, return tz-aware UTC series.

    Parameters
    ----------
    target_site : str
    full_index : DatetimeIndex
        The desired daily index. Can be naive or tz-aware.
    start_date, end_date : str
        YYYY-MM-DD bounds used for NWIS retrieval.
    parameter_code : str
        USGS parameter code (e.g., "00060", "00065", "00045", "30210"...).
    to_units : {"metric","native",None}
    tz : str
        Assumed timezone if NWIS returns naive timestamps (rare).
    parameter_kind : str or None
        Optional hint for column selection.

    Returns
    -------
    pd.Series
        Indexed by full_index (converted to UTC), filled, optionally unit-converted.
    """
    target_site = str(target_site)
    parameter_code = str(parameter_code)

    dv = nwis.get_dv(
        sites=target_site,
        parameterCd=parameter_code,
        start=start_date,
        end=end_date,
    )
    df = dv[0].copy() if isinstance(dv, (list, tuple)) else dv.copy()

    df = _ensure_datetime_index(df)
    df.index = _to_utc_index(df.index, tz=tz)

    col = _pick_parameter_col(df, parameter_code, parameter_kind=parameter_kind)
    if col is None:
        raise RuntimeError(
            f"No daily-values column found for parameter_code={parameter_code} at site={target_site}"
        )

    utc_daily = _as_utc_daily_index(full_index, tz=tz)

    ts = (
        pd.to_numeric(df[col], errors="coerce")
        .sort_index()
        .reindex(utc_daily)
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
    )

    # unit conversion (only if defined)
    ts = _convert_units_from_parameter_code(ts, parameter_code, to_units=to_units)

    ts.name = parameter_code
    return ts


def GaugebyHUC(
    start_date,
    end_date,
    huc_codes,
    parameter_code,
    percent_threshold,
    data_dir,
    json_path,
    siteType=None,
    *,
    tz="UTC",
):
    """
    Query NWIS by HUC, fetch DV, screen by completeness, and write JSON cache.

    JSON structure:
      site_dict[huc][site_no] = {
          "latitude": float,
          "longitude": float,
          "completeness_%": float,
          "parameter": { "YYYY-MM-DD": value, ... },
          "cluster": None,
          "huc": huc
      }
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    huc_codes = [str(h) for h in huc_codes]
    parameter_code = str(parameter_code)

    full_index = pd.date_range(start=start_date, end=end_date, freq="D")
    expected_days = int(len(full_index))
    utc_daily = pd.DatetimeIndex(full_index).tz_localize(tz).tz_convert("UTC")

    # Discover sites across HUCs
    site_info_list = []
    for huc in huc_codes:
        kwargs = {"huc": huc, "parameterCd": parameter_code}
        if siteType is not None:
            kwargs["siteType"] = siteType

        sites = nwis.what_sites(**kwargs)
        if isinstance(sites, tuple):
            sites = sites[0]
        if sites is None or len(sites) == 0:
            continue

        sites = sites.copy()
        sites["HUC"] = huc
        site_info_list.append(sites)

    if not site_info_list:
        raise RuntimeError("No sites returned for requested HUC codes and parameter code.")

    site_info_df = (
        pd.concat(site_info_list, ignore_index=True)
        .loc[:, ["site_no", "dec_lat_va", "dec_long_va", "HUC"]]
        .rename(
            columns={
                "dec_lat_va": "latitude",
                "dec_long_va": "longitude",
                "HUC": "huc",
            }
        )
    )

    # Pull DV and determine coverage
    data_coverage = []
    parameter_data = {}

    for site_no in site_info_df["site_no"].astype(str).tolist():
        try:
            dv = nwis.get_dv(
                sites=site_no,
                parameterCd=parameter_code,
                start=start_date,
                end=end_date,
            )
            df = dv[0].copy() if isinstance(dv, (list, tuple)) else dv.copy()
        except Exception:
            continue

        df = _ensure_datetime_index(df)
        df.index = _to_utc_index(df.index, tz=tz)

        col = _pick_parameter_col(df, parameter_code)
        if col is None:
            continue

        s = pd.to_numeric(df[col], errors="coerce").sort_index().reindex(utc_daily)

        valid_days = int(s.notna().sum())
        data_coverage.append({"site_no": site_no, "valid_days": valid_days})
        parameter_data[site_no] = s

    if not parameter_data:
        raise RuntimeError("No daily values retrieved for any site.")

    coverage_df = pd.DataFrame(data_coverage)
    coverage_df["completeness_%"] = 100.0 * coverage_df["valid_days"] / float(expected_days)
    coverage_df = (
        coverage_df.merge(site_info_df, on="site_no", how="left")
        .sort_values("completeness_%", ascending=False)
        .reset_index(drop=True)
    )

    # Apply coverage threshold
    keep = coverage_df["completeness_%"] > float(percent_threshold)
    kept_df = coverage_df.loc[keep].copy()
    kept_ids = set(kept_df["site_no"].astype(str).tolist())

    # Build nested dict by HUC
    site_dict = {h: {} for h in huc_codes}

    for site_no, s in parameter_data.items():
        site_no = str(site_no)
        if site_no not in kept_ids:
            continue

        row = kept_df.loc[kept_df["site_no"].astype(str) == site_no]
        if row.empty:
            continue
        row = row.iloc[0]
        huc = str(row["huc"])

        s2 = s.dropna()
        pairs = ((ts.date().isoformat(), float(v)) for ts, v in s2.items())

        site_dict[huc][site_no] = {
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "completeness_%": float(row["completeness_%"]),
            "parameter": dict(pairs),
            "cluster": None,
            "huc": huc,
        }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(site_dict, f)

    return {
        "num_hucs": int(len(huc_codes)),
        "num_sites_total": int(len(site_info_df)),
        "num_sites_kept": int(sum(len(v) for v in site_dict.values())),
        "json_path": str(json_path.resolve()),
    }
