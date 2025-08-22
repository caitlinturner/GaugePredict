# -*- coding: utf-8 -*-
"""
Download and cache daily-values time series from USGS NWIS by HUC.
"""

from __future__ import division, print_function, absolute_import

import json
from pathlib import Path

import pandas as pd
from dataretrieval import nwis




# --------------------------------------------------------
# Helper Functions
# --------------------------------------------------------
def _ensure_dir(path):
    """Create parent directory for a file path if missing."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _daily_index_utc(start_date, end_date):
    """Inclusive daily DatetimeIndex localized to UTC midnight."""
    return pd.date_range(start=start_date, end=end_date, freq="D").tz_localize("UTC")


def _to_utc_index(df):
    """Return df with a tz-aware UTC DatetimeIndex inferred from index or 'datetime' column."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "datetime" in out.columns:
            out.index = pd.to_datetime(out["datetime"])
        else:
            out.index = pd.to_datetime(out.index)
    out.index = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
    return out


def _rank_by_order(name, order):
    """Ranking key: first occurrence of tokens in `order`; fallback to length."""
    name_l = name.lower()
    for i, tag in enumerate(order):
        if tag in name_l:
            return i
    return len(order)


def _pick_parameter_col(df, parameter_code):
    """Choose a DV column for a USGS parameter code with sensible suffix preference.

    **Inputs** :
        df : `pd.DataFrame`
            NWIS daily-values frame.
        parameter_code : `str`
            USGS parameter code (e.g., "00060").

    **Outputs** :
        col : `str or None`
            Selected column name or None if not found.
    """
    cols = [c for c in df.columns if parameter_code in c]
    if not cols:
        return None
    order = ["mean", "sum", "value", parameter_code]
    cols_sorted = sorted(cols, key=lambda n: _rank_by_order(n, order))
    return cols_sorted[0]


# --------------------------------------------------------
# Download HUC Data
# --------------------------------------------------------
def GaugebyHUC(start_date,
               end_date,
               huc_codes,
               parameter_code,
               percent_threshold,
               data_dir,
               json_path):
    """Query NWIS by HUC, fetch daily values, screen by completeness, write JSON.

    **Inputs** :
        start_date, end_date : `str`
            Inclusive daily window (YYYY-MM-DD).
        huc_codes : `list`
            List of HUC identifiers (e.g., ["01","02"] or full HUC codes accepted by NWIS).
        parameter_code : `str`
            USGS parameter code (e.g., "00060" discharge).
        percent_threshold : `float`
            Minimum completeness percentage to keep a site (0–100).
        data_dir : `str or Path`
            Directory for any local artifacts (created if missing).
        json_path : `str or Path`
            Output JSON filepath for serialized site data.

    **Outputs** :
        summary : `dict`
            {
              "num_hucs": int,
              "num_sites_total": int,
              "num_sites_kept": int,
              "json_path": str
            }
    """
    # setup
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    json_path = Path(json_path)
    _ensure_dir(json_path)

    # expected daily index and length
    full_index_utc = _daily_index_utc(start_date, end_date)
    expected_days = len(full_index_utc)

    # discover sites across HUCs
    site_info_list = []
    for huc in huc_codes:
        sites = nwis.what_sites(huc=huc, parameterCd=parameter_code)
        if isinstance(sites, tuple):  # some versions return (df, meta)
            sites = sites[0]
        if sites is None or len(sites) == 0:
            continue
        tmp = sites.copy()
        tmp["HUC"] = huc
        site_info_list.append(tmp)

    if not site_info_list:
        raise RuntimeError("no sites returned for requested HUC codes and parameter")

    site_info_df = (
        pd.concat(site_info_list, ignore_index=True)[
            ["site_no", "dec_lat_va", "dec_long_va", "HUC"]
        ]
        .rename(columns={"dec_lat_va": "latitude", "dec_long_va": "longitude", "HUC": "huc"})
        .reset_index(drop=True)
    )

    # pull DV data and compute coverage
    data_coverage = []
    parameter_data = {}

    for site_no in site_info_df["site_no"]:
        try:
            dv = nwis.get_dv(sites=site_no,
                             parameterCd=parameter_code,
                             start=start_date,
                             end=end_date)[0]
        except Exception:
            continue  # skip sites that fail

        df = _to_utc_index(dv)
        col = _pick_parameter_col(df, parameter_code)
        if col is None:
            continue

        # align to UTC daily index before completeness
        s = df[col].sort_index().reindex(full_index_utc)
        valid_days = int(s.dropna().size)

        data_coverage.append({"site_no": site_no, "valid_days": valid_days})
        parameter_data[site_no] = s.rename("value").to_frame()

    if not parameter_data:
        raise RuntimeError("no daily values retrieved for any site")

    coverage_df = pd.DataFrame(data_coverage)
    coverage_df["completeness_%"] = 100.0 * coverage_df["valid_days"] / float(expected_days)
    coverage_df = (
        coverage_df.merge(site_info_df, on="site_no", how="left")
        .sort_values(by="completeness_%", ascending=False)
        .reset_index(drop=True)
    )

    # screen by coverage
    kept = coverage_df[coverage_df["completeness_%"] > float(percent_threshold)]
    kept_ids = set(kept["site_no"])

    # build nested dict by HUC
    site_dict = {h: {} for h in huc_codes}
    for site_no, df in parameter_data.items():
        if site_no not in kept_ids:
            continue

        # serialize as date (ISO) to float using UTC dates
        s = df["value"].dropna()
        s.index = s.index.tz_convert("UTC")
        pairs = ((ts.date().isoformat(), float(v)) for ts, v in s.items())

        row = kept.loc[kept["site_no"] == site_no].iloc[0]
        huc = row["huc"]

        site_dict[huc][site_no] = {
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "completeness_%": float(row["completeness_%"]),
            "parameter": dict(pairs),
            "cluster": None,
            "huc": huc,
        }

    # write JSON
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(site_dict, f)

    return {
        "num_hucs": int(len(huc_codes)),
        "num_sites_total": int(len(site_info_df)),
        "num_sites_kept": int(sum(len(v) for v in site_dict.values())),
        "json_path": str(json_path.resolve()),
    }
