"""
GaugePredict Data Downloader - Mississippi River Basin
Download and process USGS gauge data for the Mississippi River Basin.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
from dataretrieval import nwis
from GaugePredict.downloader import load_target, GaugebyHUC
from GaugePredict.plotting import plot_hucs, plot_statistics
from pathlib import Path

# Configuration
parameter = "discharge"
parameter_code = "00060"
target_site = "07374000"  # Baton Rouge
start_date, end_date = "2005-01-02", "2024-12-31"
units = "metric"
huc_codes = ["05", "06", "07", "08", "09", "10", "11"]
percent_threshold = 95
siteType = "ST"

# Setup directories
script_dir = Path(__file__).parent
examples_dir = script_dir.parent
data_dir = examples_dir / f"cached_data_{parameter}"
json_path = data_dir / f"site_dict_{parameter}.json"

# Create full time index
full_index = pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")

# Download target site data
print(f"Downloading target site {target_site}...")
target_df = load_target(
    target_site=target_site,
    full_index=full_index,
    parameter_code=parameter_code,
    start_date=start_date,
    end_date=end_date,
    to_units=units)

# Create continuous daily time series
target = (target_df.reindex(full_index).interpolate(limit_direction="both").ffill().bfill())

# Plot statistics for target site
fig, axes = plot_statistics(target, critical_threshold=35400.0, ylim=(0, 46000))
plt.savefig(examples_dir / "figures" / f"{target_site}_statistics.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot HUC zones with Mississippi River Basin
fig, ax = plot_hucs(
    base_dir=str(examples_dir / "shapefiles" / "HUC_Zones"),
    states_fp=str(examples_dir / "shapefiles" / "US_STATES" / "tl_2023_us_state.shp"),
    include_ak=False,
    basemap=True)

target_epsg = 4326
basin = gpd.read_file(str(examples_dir / "shapefiles" / "MSRB" / "Miss_RiverBasin.shp"))
basin = basin.to_crs(target_epsg)
basin.plot(ax=ax,
           facecolor='blue',
           edgecolor="darkblue",
           alpha=0.6,
           zorder=4)

legend = ax.legend(handles=[mpatches.Patch(facecolor='blue',
                            edgecolor="darkblue",
                            alpha=0.6,
                            label="Mississippi River Basin")],
                            frameon=False, 
                            fontsize=10)
plt.savefig(examples_dir / "figures" / "msr_basin_hucs.png", dpi=300, bbox_inches='tight')
plt.close()

# Download predictor sites by HUC (this may take a while)
print(f"Downloading predictor sites for HUC codes: {huc_codes}")

summary = GaugebyHUC(
    start_date=start_date,
    end_date=end_date,
    huc_codes=huc_codes,
    parameter_code=parameter_code,
    percent_threshold=percent_threshold,
    data_dir=data_dir,
    json_path=json_path,
    siteType=siteType)

print("\nDownload complete")
print(f"Data saved to: {data_dir}")
print(f"Site metadata saved to: {json_path}")
print(f"\nSummary: {summary}")
