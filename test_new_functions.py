# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 14:44:20 2025

@author: cturn
"""

# notebook cell
import os
import sys
sys.path.append(r"C:/Users/cturn/Documents/CSDMS/GaugePredict/GaugePredict/GaugePredict")
import geopandas as gpd
import matplotlib.patches as mpatches
import cmocean
import GaugePredict
from routines import plot_hucs 




# I want to change to this directory to run the code
os.chdir(r"C:/Users/cturn/Documents/CSDMS/GaugePredict/GaugePredict/Examples")

# Plot HUC with AK
fig, ax = plot_hucs(
    base_dir="shapefiles/HUC_Zones",
    states_fp="shapefiles/US_STATES/tl_2023_us_state.shp",
    include_ak=True,
    figsize=(8, 6),
    basemap=True)

# Plot HUC without AK
fig, ax = plot_hucs(
    base_dir="shapefiles/HUC_Zones",
    states_fp="shapefiles/US_STATES/tl_2023_us_state.shp",
    include_ak=False,
    figsize=(6, 4),
    basemap=True)



# Overlay basin on plot
fig, ax = plot_hucs(
    base_dir="shapefiles/HUC_Zones",
    states_fp="shapefiles/US_STATES/tl_2023_us_state.shp",
    include_ak=False,
    figsize=(6, 4),
    basemap=True)

basin = gpd.read_file("shapefiles/MSRB/Miss_RiverBasin.shp")
basin = basin if basin.crs.to_epsg() == 3857 else basin.to_crs(3857)
basin.plot(ax=ax, facecolor=cmocean.cm.ice(0.3), edgecolor="darkblue", alpha=0.6, zorder=4)
ax.legend(handles=[mpatches.Patch(facecolor=cmocean.cm.ice(0.3), edgecolor="darkblue", alpha=0.6,
                            label="Mississippi River Basin")],frameon=False, fontsize=14)




