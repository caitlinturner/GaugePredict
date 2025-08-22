# -*- coding: utf-8 -*-
"""
Notebook
"""
# --------------------------------------------------------
# Download data by HUCs
# --------------------------------------------------------
# Working directory (adjust as needed)
import os, sys
os.chdir(r"C:\Users\cturn\Documents\CSDMS\GaugePredict\GaugePredict")

# Ensure local package is importable (if not installed as a package)
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Core
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Geo/plots
import geopandas as gpd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
import cmocean

# USGS
from dataretrieval import nwis

# Your modules (already created)
from GaugePredict.routinesworking import (
    plot_hucs,
    load_site_json,
    build_full_index,
    build_site_series_from_json,
    compute_discharge_stats,
    plot_discharge_panels,
    load_target_series,  
    huc_average_series        
)

# HUC map (note: your plot_hucs signature does NOT take figsize; it uses internal defaults)
fig, ax = plot_hucs(
    base_dir="Examples/shapefiles/HUC_Zones",
    states_fp="Examples/shapefiles/US_STATES/tl_2023_us_state.shp",
    include_ak=False,
    basemap=True
)

# Overlay Mississippi River Basin
basin = gpd.read_file("Examples/shapefiles/MSRB/Miss_RiverBasin.shp")
basin = basin if basin.crs and basin.crs.to_epsg() == 3857 else basin.to_crs(3857)
basin.plot(ax=ax, facecolor=cmocean.cm.ice(0.3), edgecolor="darkblue", alpha=0.6, zorder=4)

legend = ax.legend(
    handles=[mpatches.Patch(facecolor=cmocean.cm.ice(0.3), edgecolor="darkblue",
                            alpha=0.6, label="Mississippi River Basin")],
    frameon=False, fontsize=12, loc="lower left"
)


# Analysis window and units
START_DATE = "2005-01-01"
END_DATE   = "2024-12-31"
TZ         = "UTC"

PARAMETER        = "discharge"      # maps to 00060 in your routines
ASSUME_INPUT     = "customary"      # cached JSON typically in cfs
TARGET_UNITS     = "metric"         # convert to m^3/s

# Paths
PROJECT_DIR = Path.cwd()
DATA_DIR    = PROJECT_DIR / "Examples" / "cached_data_discharge"
JSON_PATH   = r"C:\Users\cturn\Documents\CSDMS\GaugePredict\data\cached_data_discharge\site_dict_discharge.json"    # adjust if name differs
OUT_DIR     = PROJECT_DIR / "outputs" / "gaugebyhuc_discharge"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (Optional) HUC list for documentation only; not needed if JSON already exists
huc_codes = ["04", "05", "06", "07", "08", "09", "10", "11", "14", "17"]



parameter = "discharge"
parameter_code = "00060"
target_site = "07374000"
start_date, end_date = "2005-01-02", "2024-12-31"

df = nwis.get_dv(sites=target_site, parameterCd=parameter_code,
                 start=start_date, end=end_date)[0]

target = df["00060_Mean"] * 0.0283168466
target.index = pd.to_datetime(target.index).tz_localize("UTC") if target.index.tz is None else target.index.tz_convert("UTC")

filled = (target.reindex(pd.date_range(start=start_date, end=end_date))
                .tz_localize("UTC")
                .interpolate(limit_direction="both")
                .ffill().bfill())

stats, s, v, q = compute_discharge_stats(target, start_date, end_date, tz="UTC")
fig, axes = plot_discharge_panels(s, v, q, opening_threshold=35400.0, y_max=46000)
fig.suptitle("Target Site: Baton Rouge", y=1.05, fontsize=10)

# Load cached JSON (written previously by GaugebyHUC)
raw = load_site_json(JSON_PATH)

# Daily index (timezone-aware)
full_index = build_full_index(START_DATE, END_DATE, tz=TZ)

# Build aligned, unit-converted per-site series from JSON
site_series = build_site_series_from_json(
    raw=raw,
    full_index=full_index,
    parameter=PARAMETER,
    to_units=TARGET_UNITS,          # cfs -> m^3/s
    assume_input_units=ASSUME_INPUT,
    series_key="parameter",         # matches your GaugebyHUC writer
    alt_keys=("discharge", "water_level", "stage", "precipitation"),
)



# Long table: site_no, date, discharge_m3s
records = []
for site_no, s in site_series.items():
    df = pd.DataFrame({"date": s.index.normalize(), "discharge_m3s": s.values})
    df["site_no"] = site_no
    records.append(df)

df_long = (pd.concat(records, ignore_index=True)
             .assign(date=lambda d: pd.to_datetime(d["date"]))
             .sort_values(["site_no", "date"]))

# Metadata from JSON (HUC, lat/long, completeness)
meta_rows = []
for huc, hsites in raw.items():
    for site_no, info in hsites.items():
        meta_rows.append({
            "site_no": site_no,
            "huc": info.get("huc", huc),
            "latitude": info.get("latitude", np.nan),
            "longitude": info.get("longitude", np.nan),
            "completeness_pct": info.get("completeness_%", np.nan),
        })
meta = pd.DataFrame(meta_rows).drop_duplicates("site_no")

# Merge daily with metadata and write
df_long_meta = df_long.merge(meta, on="site_no", how="left")


# Site-level descriptive statistics
site_stats = []
for site_no, s in site_series.items():
    stats, s_aligned, v_nonan, q = compute_discharge_stats(
        target=s, start_date=START_DATE, end_date=END_DATE, tz=TZ
    )
    stats["site_no"] = site_no
    site_stats.append(stats)

df_site_stats = (pd.concat(site_stats, ignore_index=True)
                   .merge(meta, on="site_no", how="left"))

# HUC-level aggregates (mean of quantiles; site count)
agg = {
    "mean":      ["mean", "median"],
    "q05":       "mean",
    "q25":       "mean",
    "median":    "mean",
    "q75":       "mean",
    "q95":       "mean",
    "coverage_%":"mean",
    "latitude":  "count",   # site count
}
by_huc = (df_site_stats
          .groupby("huc", dropna=False)
          .agg(agg))
by_huc.columns = ["_".join([c for c in col if c]) for col in by_huc.columns.to_flat_index()]
by_huc = by_huc.rename(columns={"latitude_count": "n_sites"}).reset_index()





# discover all HUCs present in the cache
all_hucs = sorted({str(info.get("huc", h)) for h, hsites in raw.items() for info in hsites.values()})

agg_dir = OUT_DIR / "huc_aggregates_mean"
agg_dir.mkdir(parents=True, exist_ok=True)

rows = []
for huc_code in all_hucs:
    try:
        s_huc = huc_average_series(huc_code, site_series, meta, full_index, agg="mean")
    except ValueError:
        continue

    # stats and figure
    stats, s_aligned, v_nonan, q = compute_discharge_stats(
        target=s_huc, start_date=START_DATE, end_date=END_DATE, tz=TZ
    )
    fig, axes = plot_discharge_panels(s=s_aligned, v=v_nonan, q=q,
                                      opening_threshold=None, y_max=None)
    fig.suptitle(f"HUC {huc_code}", y=1.05, fontsize=10)

    # collect summary row
    r = stats.copy()
    r["huc"] = huc_code
    rows.append(r)




rows = []
for huc_code in all_hucs:
    try:
        s_huc = huc_average_series(huc_code, site_series, meta, full_index, agg="sum")
    except ValueError:
        continue

    # stats and figure
    stats, s_aligned, v_nonan, q = compute_discharge_stats(
        target=s_huc, start_date=START_DATE, end_date=END_DATE, tz=TZ
    )
    fig, axes = plot_discharge_panels(s=s_aligned, v=v_nonan, q=q,
                                      opening_threshold=None, y_max=None)
    fig.suptitle(f"HUC {huc_code}", y=1.05, fontsize=10)

    # collect summary row
    r = stats.copy()
    r["huc"] = huc_code
    rows.append(r)
