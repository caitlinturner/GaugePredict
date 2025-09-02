# downloader_test.py — simple version

import json
from pathlib import Path
import pandas as pd
from dataretrieval import nwis


def _pick_parameter_col(df, parameter_code):
    """
    Choose the DV column for a USGS parameter code.
    """
    cols = [c for c in df.columns if parameter_code in c]
    if not cols:
        return None

    order = ["mean", "sum", "value", parameter_code]

    def rank(name):
        name_l = name.lower()
        for i, tag in enumerate(order):
            if tag in name_l:
                return i
        return len(order)

    cols_sorted = sorted(cols, key=rank)
    return cols_sorted[0]


def GaugebyHUC(
    start_date,
    end_date,
    huc_codes,
    parameter_code,
    percent_threshold,
    data_dir,
    json_path,
):
    """
    Query NWIS sites by HUC, fetch daily values for a parameter, screen by coverage,
    and write JSON.

    Returns a summary dict with counts and file path.
    """
    # setup
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # expected daily index and length
    full_index = pd.date_range(start=start_date, end=end_date, freq="D")  # inclusive
    expected_days = len(full_index)

    # discover sites
    site_info_list = []
    for huc in huc_codes:
        sites = nwis.what_sites(huc=huc, parameterCd=parameter_code)
        if isinstance(sites, tuple):  # some versions return (df, meta)
            sites = sites[0]
        if sites is None or len(sites) == 0:
            continue
        sites = sites.copy()
        sites["HUC"] = huc
        site_info_list.append(sites)

    if not site_info_list:
        raise RuntimeError("no sites returned for requested HUC codes and parameter")

    site_info_df = (
        pd.concat(site_info_list, ignore_index=True)[["site_no", "dec_lat_va", "dec_long_va", "HUC"]]
        .rename(columns={"dec_lat_va": "latitude", "dec_long_va": "longitude", "HUC": "huc"})
    )

    # pull DV data
    data_coverage = []
    parameter_data = {}

    for site_no in site_info_df["site_no"]:
        try:
            dv = nwis.get_dv(sites=site_no, parameterCd=parameter_code, start=start_date, end=end_date)[0]
        except Exception:
            continue  # skip sites that fail

        df = dv.copy()

        # normalize index to tz-aware UTC
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df.index = pd.to_datetime(df["datetime"])
            else:
                df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        col = _pick_parameter_col(df, parameter_code)
        if col is None:
            continue

        # align to full daily index (as UTC midnight) before counting completeness
        # make a UTC-midnight daily index to match full_index dates
        utc_daily = pd.DatetimeIndex(full_index.tz_localize("UTC"))
        s = (
            df[col]
            .sort_index()
            .reindex(utc_daily)
        )

        valid_days = s.dropna().size
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
    high_quality = coverage_df[coverage_df["completeness_%"] > percent_threshold]
    high_ids = set(high_quality["site_no"])

    # build nested dict by HUC
    site_dict = {h: {} for h in huc_codes}
    for site_no, df in parameter_data.items():
        if site_no not in high_ids:
            continue

        # serialize as date (ISO) -> float; ensure UTC then take date
        s = df["value"].dropna()
        s.index = s.index.tz_convert("UTC")
        pairs = ((ts.date().isoformat(), float(v)) for ts, v in s.items())

        row = high_quality.loc[high_quality["site_no"] == site_no].iloc[0]
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
        "num_hucs": len(huc_codes),
        "num_sites_total": int(len(site_info_df)),
        "num_sites_kept": int(sum(len(v) for v in site_dict.values())),
        "json_path": str(json_path.resolve()),
    }


