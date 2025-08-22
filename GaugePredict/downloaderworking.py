# -*- coding: utf-8 -*-
"""
Download and cache daily-values time series from USGS NWIS by HUC.
downloaderworking.py
"""
from __future__ import division, print_function, absolute_import

import json
from pathlib import Path
import pandas as pd
from dataretrieval import nwis

# --------------------------------------------------------
# Download data by HUCs
# --------------------------------------------------------

def GaugebyHUC(start_date,
               end_date,
               huc_codes,
               parameter_code,
               percent_threshold,
               data_dir,
               json_path):
    """
    Query NWIS by HUC, fetch daily values, screen by completeness, write JSON.

    Inputs:
        start_date, end_date : str
            Inclusive daily window (YYYY-MM-DD).
        huc_codes : list
            List of HUC identifiers (e.g., ["01","02"] or full HUC codes).
        parameter_code : str
            USGS parameter code (e.g., "00060" for discharge).
        percent_threshold : float
            Minimum completeness percentage to keep a site (0–100).
        data_dir : str or Path
            Directory for local artifacts (created if missing).
        json_path : str or Path
            Output JSON filepath for serialized site data.

    Outputs:
        dict:
            {
              "num_hucs": int,
              "num_sites_total": int,
              "num_sites_kept": int,
              "json_path": str
            }
    """
    # Setup
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Inclusive UTC daily index and expected length
    full_index_utc = pd.date_range(start=start_date, end=end_date, freq="D").tz_localize("UTC")
    expected_days = len(full_index_utc)

    # Discover sites across HUCs
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

    # Pull DV data and compute coverage
    data_coverage = []
    parameter_data = {}

    for site_no in site_info_df["site_no"]:
        try:
            dv = nwis.get_dv(
                sites=site_no,
                parameterCd=parameter_code,
                start=start_date,
                end=end_date
            )
            dv = dv[0] if isinstance(dv, (list, tuple)) else dv
        except Exception:
            continue  # skip sites that fail

        # Ensure DatetimeIndex and convert to UTC
        df = dv.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df.index = pd.to_datetime(df["datetime"])
            else:
                df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

        # Pick parameter column with suffix preference
        candidates = [c for c in df.columns if parameter_code in c]
        if not candidates:
            continue
        order = ["mean", "sum", "value", parameter_code]
        def _rank(name: str) -> int:
            name_l = name.lower()
            for i, tag in enumerate(order):
                if tag in name_l:
                    return i
            return len(order)
        col = sorted(candidates, key=_rank)[0]

        # Align to UTC daily index before completeness
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

    # Screen by coverage
    kept = coverage_df[coverage_df["completeness_%"] > float(percent_threshold)]
    kept_ids = set(kept["site_no"])

    # Build nested dict by HUC
    site_dict = {h: {} for h in huc_codes}
    for site_no, df in parameter_data.items():
        if site_no not in kept_ids:
            continue

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

    # Write JSON
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(site_dict, f)

    return {
        "num_hucs": int(len(huc_codes)),
        "num_sites_total": int(len(site_info_df)),
        "num_sites_kept": int(sum(len(v) for v in site_dict.values())),
        "json_path": str(json_path.resolve()),
    }
