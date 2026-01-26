# -*- coding: utf-8 -*-
"""
GaugePredict/downloader.py

Utilities for retrieving USGS NWIS daily-values (DV) time series and assembling
a screened gauge catalog by Hydrological Unit Code (HUC).

"""

from __future__ import division, print_function, absolute_import

import json
from pathlib import Path

import numpy as np
import pandas as pd
from dataretrieval import nwis


# =============================================================================
# Data selection utils
# =============================================================================

def _rank_parameter_col(name, preference):
    """
        Rank a candidate NWIS daily-values column name based on keyword preference.
        
        **Inputs** :
        
            name : 'str'
                Column name to rank.
        
            preference : 'list of str'
                Ordered list of substrings to search for in the column name.
        
        **Outputs** :
        
            rank : 'int'
                Rank index in `preference` for the first match; returns len(preference)
                if no preference tag is found.
    """
    name_l = str(name).lower()
    for i, tag in enumerate(preference):
        if tag in name_l:
            return i
    return len(preference)


def _pick_parameter_col(df, parameter_code, *, parameter_kind=None):
        """
        Select the preferred NWIS DV column corresponding to a parameter code.
    
        This function filters columns that include `parameter_code` in their name,
        then selects the best match using a simple preference order.
        For precipitation-like parameters, "sum/total/accum" is prioritized since
        daily totals are commonly desired. For other parameters, "mean/value" is
        prioritized.
    
        **Inputs** :
    
            df : 'pandas.DataFrame'
                Daily-values table returned by `dataretrieval.nwis.get_dv()`.
    
            parameter_code : 'str or int'
                USGS parameter code used to identify candidate columns.
    
            parameter_kind : 'str or None'
                Optional ranking identification hint. Expected values include
                "discharge", "water_level", "precipitation" (or "precip").
    
        **Outputs** :
    
            col : 'str or None'
                Name of the selected column, or None if no matching column exists.
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
    """
        Ensure a DataFrame is indexed by pandas.DatetimeIndex.
    
        Allows flexibility in dataframes, as dataframes that are sometimes already indexed by datetime, and
        sometimes include a "datetime" column. This helper normalizes both cases.
    
        **Inputs** :
    
            df : 'pandas.DataFrame'
                NWIS DV table.
    
        **Outputs** :
    
            df_out : 'pandas.DataFrame'
                Copy of `df` with a DatetimeIndex.
    """
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
        Convert an index to a tz-aware UTC pandas.DatetimeIndex.
    
        If `idx` is timezone-naive, it is localized to `tz` first, then converted
        to UTC. If `idx` already has timezone information, it is directly converted
        to UTC.
    
        **Inputs** :
    
            idx : 'array-like of datetime-like'
                Index or datetime-like values convertible by pandas.to_datetime().
    
            tz : 'str'
                Timezone to assume when `idx` is timezone-naive.
    
        **Outputs** :
    
            idx_utc : 'pandas.DatetimeIndex'
                Timezone-aware DatetimeIndex in UTC.
    """
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize(tz)
    return idx.tz_convert("UTC")


def _as_utc_daily_index(full_index, *, tz="UTC"):
    """
        Convert a provided daily index to a tz-aware UTC pandas.DatetimeIndex.
    
        This helper is intended for a user-provided "full_index" (i.e., uploaded .txt or csv.) defining desired
        day stamps. If the index is timezone-naive, it is localized to `tz` and then
        converted to UTC.
    
        **Inputs** :
    
            full_index : 'array-like of datetime-like'
                Daily timestamps (naive or tz-aware).
    
            tz : 'str'
                Timezone to assume when `full_index` is timezone-naive.
    
        **Outputs** :
    
            idx_utc : 'pandas.DatetimeIndex'
                Daily DatetimeIndex, timezone-aware in UTC.
    """
    idx = pd.to_datetime(full_index)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize(tz)
    return idx.tz_convert("UTC")


def _convert_units_from_parameter_code(ts, parameter_code, *, to_units="metric"):
    """
        Convert common USGS DV units to metric using parameter code conventions.
    
        This implements a small set of explicit conversions for commonly used NWIS
        daily values. Additions are welcome for your use case:
            - 00060 discharge: cfs -> m^3/s
            - 00065 gage height / stage: ft -> m
            - 00045 precipitation: inches -> mm
    
        If `to_units` is "native" or None, the series is returned unchanged.
    
        **Inputs** :
    
            ts : 'pandas.Series'
                Time series of daily values.
    
            parameter_code : 'str or int'
                USGS parameter code used to choose conversion.
    
            to_units : {'metric', 'native', None}
                Target unit system. For now, only "metric" is supported for conversion.
    
        **Outputs** :
    
            ts_out : 'pandas.Series'
                Series converted to metric units, or unchanged if native requested.
    
        **Raises** :
    
            ValueError
                If `to_units` is unsupported, or if no conversion is defined for
                the provided parameter code.
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
# Fetching USGS Data 
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
    Retrieve NWIS daily-values for a site and align to a provided daily index.

    This function downloads daily values (DV) from USGS NWIS for a single site
    and parameter, converts the series to a timezone-aware UTC index, and then
    reindexes to `full_index` (daily). Missing days are filled by interpolation
    and edge filling (ffill/bfill). Optionally converts units to metric for a
    small set of common parameter codes using helper functions withing downloader.py.

    **Inputs** :
            target_site : 'str'
                USGS site identifier (e.g., "06730200").
    
            full_index : 'pandas.DatetimeIndex'
                Desired daily index to align the returned series to. Can be timezone
                naive or timezone aware.
    
            start_date : 'str'
                Retrieval start date in "YYYY-MM-DD" format.
    
            end_date : 'str'
                Retrieval end date in "YYYY-MM-DD" format.
    
            parameter_code : 'str'
                USGS parameter code (e.g., "00060" discharge, "00065" stage,
                "00045" precipitation).
    
            to_units : {'metric', 'native', None}
                If "metric", apply code-based unit conversions where defined.
                If "native" or None, return native units.
    
            tz : 'str'
                Timezone used to localize timestamps when NWIS returns timezone-naive
                datetimes (or when `full_index` is timezone-naive).
    
            parameter_kind : 'str or None'
                Optional hint to improve column selection when multiple DV columns
                exist (e.g., precipitation prefers sum/total).
    
        **Outputs** :
    
            ts : 'pandas.Series'
                Daily series indexed by `full_index` converted to UTC, gap-filled,
                and optionally converted to metric units. The series name is set to
                `parameter_code`.
    
        **Raises** :
    
            RuntimeError
                If no matching DV column is found for the requested parameter code.
    
            ValueError
                If an unsupported unit conversion is requested.
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
        Build a HUC-grouped gauge catalog from NWIS, screened by data completeness.
    
        For each requested HUC code, this function:
            1) Queries NWIS site metadata for the given parameter code.
            2) Downloads daily values (DV) for each site for the requested date range.
            3) Computes data completeness as (non-NaN days / expected days) * 100.
            4) Keeps only sites above `percent_threshold`.
            5) Writes a JSON cache keyed by HUC then site number.
    
        JSON structure:
    
            site_dict[huc][site_no] = {
                "latitude": float,
                "longitude": float,
                "completeness_%": float,
                "parameter": { "YYYY-MM-DD": value, ... },
                "cluster": None,
                "huc": huc
            }
    
    
        **Inputs** :
    
            start_date : 'str'
                Retrieval start date in "YYYY-MM-DD" format.
    
            end_date : 'str'
                Retrieval end date in "YYYY-MM-DD" format.
    
            huc_codes : 'list of str or list of int'
                One or more HUC identifiers to query.
    
            parameter_code : 'str'
                USGS parameter code used for site discovery and DV retrieval.
    
            percent_threshold : 'float'
                Completeness threshold in percent. Sites with completeness strictly
                greater than this value are kept.
    
            data_dir : 'str or pathlib.Path'
                Directory path created if it does not exist. Included for workflow
                compatibility across OSes.
    
            json_path : 'str or pathlib.Path'
                Output JSON file path. Parent directories are created as needed.
    
            siteType : 'str or None'
                Optional NWIS siteType filter (passed through to nwis.what_sites()).
    
            tz : 'str'
                Timezone used to localize naive timestamps (and the daily date range)
                before converting to UTC.
    
        **Outputs** :
    
            summary : 'dict'
                Dictionary summarizing results with keys:
                    - "num_hucs": int
                    - "num_sites_total": int
                    - "num_sites_kept": int
                    - "json_path": str (resolved absolute path)
    
        **Raises** :
    
            RuntimeError
                If no sites are returned for the requested HUC codes and parameter,
                or if no daily values are retrieved for any site.
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
