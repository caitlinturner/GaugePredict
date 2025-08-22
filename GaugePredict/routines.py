# -*- coding: utf-8 -*-
"""
Utilities for loading HUC data, processing gauge series, and plotting hydrologic figures.
"""

from __future__ import division, print_function, absolute_import

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import contextily as ctx
import cmocean as cm

from dataretrieval import nwis


# --------------------------------------------------------
# Figure defaults (for a paper)
# --------------------------------------------------------
default_figsize = (7.33, 3.54)   
default_dpi = 400
conus_extent = (-14_000_000, -7_300_000, 2_600_000, 7_000_000)
conus_ak_extent = (-20_000_000, -7_000_000, 2_600_000, 11_700_000)

# shared plotting palettes
def _palette():
    """Return shared color/linestyle sequences to avoid repetition."""
    colors = [cm.cm.haline(x) for x in (0.05, 0.20, 0.40, 0.55, 0.65, 0.75)]
    linestyles = ['--', '-.', ':', '--', '-.', ':']
    return colors, linestyles


# --------------------------------------------------------
# Helper Functions
# --------------------------------------------------------
def figax(figsize=default_figsize, dpi=default_dpi):
    """Create a figure and axis with standard labeling.

    **Inputs** :
        figsize : `tuple`
            Figure size in inches.
        dpi : `int`
            Figure dpi.

    **Outputs** :
        fig, ax : `matplotlib.figure.Figure`, `matplotlib.axes.Axes`
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    return fig, ax


def _read_3857(fp):
    """Read a vector file and return Web Mercator (EPSG:3857)."""
    gdf = gpd.read_file(fp)
    if gdf.crs is None:
        raise ValueError(str(fp) + " has no CRS")
    return gdf if gdf.crs.to_epsg() == 3857 else gdf.to_crs(3857)


def _set_extent(ax, extent):
    """Apply map extent."""
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def add_gray_basemap(ax, zoom=4):
    """Add a neutral gray basemap under existing content."""
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas,
                    attribution=False, zoom=zoom, crs=3857)


def _union_geom(gdf):
    """Union polygons robustly across geopandas versions."""
    return gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union


# --------------------------------------------------------
# HUC discovery / loading
# --------------------------------------------------------
def _discover_huc_codes(base_dir):
    """Find available HUC2 codes under a directory.

    **Inputs** :
        base_dir : `str`
            Directory with subfolders HUC##/WBDHU2.shp.

    **Outputs** :
        codes : `list`
            Sorted list of HUC2 strings.
    """
    hits = list(Path(base_dir).glob("HUC??/WBDHU2.shp"))
    if not hits:
        raise FileNotFoundError("no HUC shapefiles under %s" % base_dir)
    return sorted({p.parent.name[-2:] for p in hits})


def load_hucs_3857(base_dir):
    """Load all HUC2 polygons to EPSG:3857 with normalized columns.

    **Inputs** :
        base_dir : `str`
            Directory containing HUC## folders and WBDHU2.shp.

    **Outputs** :
        gdf : `geopandas.GeoDataFrame`
            HUC2 polygons in EPSG:3857 with column 'huc2'.
    """
    codes = _discover_huc_codes(base_dir)
    gdfs = []
    for code in codes:
        shp = Path(base_dir) / ("HUC%s" % code) / "WBDHU2.shp"
        if not shp.exists():
            warnings.warn("missing: %s (skipping)" % shp)
            continue
        df = gpd.read_file(shp).rename(columns=str.lower)
        if "huc2" not in df.columns:
            if "huc02" in df.columns:
                df = df.rename(columns={"huc02": "huc2"})
            else:
                df = df.assign(huc2=code)
        df["huc2"] = df["huc2"].astype(str).str.zfill(2)
        gdfs.append(df)
    if not gdfs:
        raise FileNotFoundError("no usable HUC shapefiles after scanning")
    out = pd.concat(gdfs, ignore_index=True)
    return out if out.crs and out.crs.to_epsg() == 3857 else out.to_crs(3857)


def simple_huc_colors(codes):
    """Map HUC codes to colors using tab20, repeating as needed."""
    base = list(plt.colormaps["tab20"].colors)
    k = len(codes)
    palette = (base * int(np.ceil(k / len(base))))[:k]
    return {c: palette[i] for i, c in enumerate(sorted(codes))}


# --------------------------------------------------------
# Plotting: HUC maps
# --------------------------------------------------------
def plot_hucs(base_dir,
              states_fp,
              include_ak=False,
              label_hucs=True,
              basemap=True,
              zoom=4):
    """Plot HUC2 polygons with state boundaries.

    **Inputs** :
        base_dir : `str`
            Directory with HUC##/WBDHU2.shp.
        states_fp : `str`
            State boundary shapefile or geopackage.
        include_ak : `bool`
            If True, include Alaska and larger extent.
        label_hucs : `bool`
            Draw HUC2 labels at representative points.
        basemap : `bool`
            Include gray basemap.
        zoom : `int`
            Basemap zoom.

    **Outputs** :
        fig, ax : `matplotlib` figure and axes
    """
    basins = load_hucs_3857(base_dir)
    states = _read_3857(states_fp).rename(columns=str.lower)

    if include_ak:
        states_plot = states
        basins_plot = basins
        extent = conus_ak_extent
    else:
        states_plot = states[~states["stusps"].isin(["AK", "HI", "PR", "VI"])]
        basins_plot = basins[basins["huc2"] != "19"]
        extent = conus_extent

    mapping = simple_huc_colors(sorted(basins_plot["huc2"].unique()))
    basins_plot = basins_plot.assign(_color=basins_plot["huc2"].map(mapping))

    fig, ax = figax(figsize=default_figsize, dpi=default_dpi)
    basins_plot.plot(ax=ax, facecolor=basins_plot["_color"],
                     edgecolor="dimgray", linewidth=0.4, alpha=0.30, zorder=1)
    states_plot.boundary.plot(ax=ax, linewidth=0.5, edgecolor="gray",
                              alpha=0.6, zorder=2)

    if label_hucs:
        for h, sub in basins_plot.groupby("huc2"):
            rp = _union_geom(sub).representative_point()
            ax.text(rp.x, rp.y, h, ha="center", va="center",
                    fontsize=8, fontweight="bold", zorder=3)

    if basemap:
        add_gray_basemap(ax, zoom=zoom)

    _set_extent(ax, extent)
    return fig, ax


# --------------------------------------------------------
# Gauge tools: units, parsing, conversion
# --------------------------------------------------------
# units: customary -> (cfs, ft, in); metric -> (m^3/s, m, cm)
_unit_converters = {
    "discharge": {
        ("customary", "metric"):    lambda s: s * 0.0283168466,   # cfs -> m³/s
        ("metric",    "customary"): lambda s: s / 0.0283168466,
    },
    "water_level": {
        ("customary", "metric"):    lambda s: s * 0.3048,         # ft -> m
        ("metric",    "customary"): lambda s: s / 0.3048,
    },
    "precipitation": {
        ("customary", "metric"):    lambda s: s * 2.54,           # in -> cm
        ("metric",    "customary"): lambda s: s / 2.54,
    },
}

_param_defaults = {
    "discharge":     "00060",
    "water_level":   "00065",
    "precipitation": "00045",
}


def build_full_index(start_date, end_date, tz="UTC"):
    """Daily datetime index with timezone.

    **Inputs** :
        start_date, end_date : `str`
            Inclusive bounds for daily frequency.
        tz : `str`
            Timezone key.

    **Outputs** :
        idx : `pd.DatetimeIndex`
    """
    idx = pd.date_range(start=start_date, end=end_date, freq="D")
    return idx.tz_localize(tz) if getattr(idx, "tz", None) is None else idx.tz_convert(tz)


def _to_series(ts):
    """Coerce to numeric Series with datetime index."""
    s = pd.Series(ts) if not isinstance(ts, pd.Series) else ts.copy()
    s.index = pd.to_datetime(s.index)
    s = pd.to_numeric(s, errors="coerce")
    return s


def _convert_units(s, parameter, from_units, to_units):
    """Unit conversion wrapper with parameter-specific mappings."""
    parameter = str(parameter).lower()
    key = (str(from_units).lower(), str(to_units).lower())
    if from_units == to_units:
        return s
    if parameter not in _unit_converters or key not in _unit_converters[parameter]:
        raise ValueError("unsupported conversion: parameter=%s, %s->%s" %
                         (parameter, from_units, to_units))
    return _unit_converters[parameter][key](s)


def process_series(ts,
                   index,
                   parameter,
                   to_units="metric",
                   assume_input_units="customary",
                   force_tz="UTC",
                   sentinels=(-999999, -99999, -9999),
                   fill=True):
    """Parse → TZ normalize → align to index → convert units → optional fill.

    **Inputs** :
        ts : `Series or array-like`
            Time series keyed by datetime index.
        index : `pd.DatetimeIndex`
            Target index to reindex against.
        parameter : `str`
            'discharge' | 'water_level' | 'precipitation'
        to_units : `str`
            'metric' or 'customary'
        assume_input_units : `str`
            Units assumed for input series.
        force_tz : `str`
            Timezone for index alignment.
        sentinels : `tuple`
            Values to treat as missing.
        fill : `bool`
            If True, interpolate/ffill/bfill.

    **Outputs** :
        s : `pd.Series`
            Processed series on `index`.
    """
    s = _to_series(ts).replace(list(sentinels), np.nan)

    if s.index.tz is None:
        s.index = s.index.tz_localize(force_tz)
    else:
        s.index = s.index.tz_convert(force_tz)
    idx = index if getattr(index, "tz", None) is not None else index.tz_localize(force_tz)

    s = s.sort_index().reindex(idx)
    s = _convert_units(s, parameter=parameter,
                       from_units=assume_input_units, to_units=to_units)

    if fill:
        s = s.interpolate(limit_direction="both").ffill().bfill()
    return s


def load_site_json(json_path):
    """Load a JSON file (UTF-8)."""
    with Path(json_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_site_series_from_json(raw,
                                full_index,
                                parameter,
                                to_units="metric",
                                assume_input_units="customary",
                                series_key="parameter",
                                alt_keys=("discharge", "water_level", "stage", "precipitation")):
    """Build site series dict from cached JSON.

    **Inputs** :
        raw : `dict`
            Nested JSON (HUC -> {site_no: info}).
        full_index : `DatetimeIndex`
            Target daily index.
        parameter : `str`
            'discharge'|'water_level'|'precipitation'
        to_units, assume_input_units, series_key, alt_keys : see above.

    **Outputs** :
        out : `dict`
            {site_no: Series}
    """
    out = {}
    for huc_sites in raw.values():
        for site_no, info in huc_sites.items():
            found = info.get(series_key, None)
            if found is None:
                for k in alt_keys:
                    if k in info:
                        found = info[k]
                        break
            if found is None:
                continue
            s = process_series(found, full_index,
                               parameter=parameter,
                               to_units=to_units,
                               assume_input_units=assume_input_units)
            out[site_no] = s
    return out


def _pick_dv_column(df, parameter_code, parameter):
    """Choose a DV column for a USGS parameter code, with suffix preference."""
    cols = [c for c in df.columns if parameter_code in c]
    if not cols:
        return None
    prefer = ("sum", "total", "accum") if parameter == "precipitation" else ("mean", "value")
    def rank(name):
        name_l = name.lower()
        for i, tag in enumerate(prefer):
            if tag in name_l:
                return i
        return len(prefer)
    return sorted(cols, key=rank)[0]


def load_target_series(target_site,
                       start_date,
                       end_date,
                       parameter,
                       parameter_code=None,
                       to_units="metric",
                       assume_input_units="customary",
                       force_tz="UTC"):
    """Download a target DV series and return daily, filled, converted.

    **Inputs** :
        target_site : `str`
            USGS site id.
        start_date, end_date : `str`
            Date bounds for DV query.
        parameter : `str`
            'discharge' | 'water_level' | 'precipitation'
        parameter_code : `str`, optional
            Override NWIS code; defaults provided internally.
        to_units, assume_input_units, force_tz : see above.

    **Outputs** :
        s : `pd.Series`
            Daily series on full index, filled and unit-converted.
    """
    parameter = parameter.lower()
    code = parameter_code or _param_defaults.get(parameter)
    if code is None:
        raise ValueError("no default parameter code for parameter='%s'" % parameter)

    idx = build_full_index(start_date, end_date, tz=force_tz)
    dv = nwis.get_dv(sites=target_site, parameterCd=code, start=idx[0], end=idx[-1])[0]

    col = _pick_dv_column(dv, code, parameter)
    if col is None:
        raise RuntimeError("no DV column found for code %s at site %s" % (code, target_site))

    s = dv[col]
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize(force_tz)
    else:
        s.index = s.index.tz_convert(force_tz)

    s = s.sort_index().reindex(idx).interpolate(limit_direction="both").ffill().bfill()
    s = _convert_units(s, parameter=parameter,
                       from_units=assume_input_units, to_units=to_units)
    return s


def stack_features(site_series_by_site,
                   target_series,
                   include_target=True,
                   include_target_diff=True):
    """Stack channels into [n_channels, n_times] array with names.

    **Inputs** :
        site_series_by_site : `dict`
            {site_no: Series}
        target_series : `Series`
            Aligned to full index.
        include_target : `bool`
            Append target channel.
        include_target_diff : `bool`
            Append first-difference channel.

    **Outputs** :
        x_raw : `ndarray`
        y : `ndarray`
        names : `list`
    """
    feats, names = [], []

    for site_no, s in site_series_by_site.items():
        feats.append(s.reindex_like(target_series))
        names.append("site_%s" % site_no)

    if include_target:
        feats.append(target_series)
        names.append("target")

    if include_target_diff:
        feats.append(target_series.diff().bfill())
        names.append("target_diff")

    x_raw = np.stack([s.values for s in feats], axis=0)
    y = target_series.values.copy()
    return x_raw, y, names


# --------------------------------------------------------
# Discharge: statistics + panels
# --------------------------------------------------------
def compute_discharge_stats(target,
                            start_date,
                            end_date,
                            tz=None):
    """Compute summary statistics and aligned daily series.

    **Inputs** :
        target : `pd.Series`
            Discharge series with tz-aware index.
        start_date, end_date : `str`
            Inclusive daily bounds.
        tz : `str`, optional
            Output timezone; if None, inferred from target.

    **Outputs** :
        stats : `pd.DataFrame`
            One row of summary stats.
        s : `pd.Series`
            Daily reindexed (no interpolation).
        v : `pd.Series`
            Valid values (dropna).
        q : `pd.Series`
            Quantiles at [0.05, 0.25, 0.50, 0.75, 0.95].
    """
    tz_out = tz or getattr(getattr(target.index, "tz", None), "key", "UTC") or "UTC"
    idx = pd.date_range(start=start_date, end=end_date, freq="D", tz=tz_out)
    s = target.sort_index().tz_convert(tz_out).reindex(idx)

    expected_n = len(idx)
    n_valid = int(s.notna().sum())
    n_missing = expected_n - n_valid
    coverage_pct = 100.0 * n_valid / expected_n if expected_n else np.nan

    v = s.dropna()
    q = v.quantile([0.05, 0.25, 0.50, 0.75, 0.95]) if len(v) else pd.Series(dtype=float)

    stats = pd.DataFrame([{
        "start": s.index.min(),
        "end": s.index.max(),
        "expected_n": expected_n,
        "n_valid": n_valid,
        "n_missing": n_missing,
        "coverage_%": coverage_pct,
        "mean": float(v.mean()) if len(v) else np.nan,
        "std": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
        "min": float(v.min()) if len(v) else np.nan,
        "q05": float(q.get(0.05, np.nan)),
        "q25": float(q.get(0.25, np.nan)),
        "median": float(q.get(0.50, np.nan)),
        "q75": float(q.get(0.75, np.nan)),
        "q95": float(q.get(0.95, np.nan)),
        "max": float(v.max()) if len(v) else np.nan
    }])

    return stats, s, v, q


def plot_discharge_panels(s,
                          v,
                          q,
                          opening_threshold=None,
                          y_max=None):
    """Three-panel discharge figure: time series, histogram, monthly violins.

    **Inputs** :
        s : `pd.Series`
            Daily series incl. NaN (for x-axis span).
        v : `pd.Series`
            Valid-only values.
        q : `pd.Series`
            Quantile series (index in [0,1]).
        opening_threshold : `float`, optional
            Reference line for flows.
        y_max : `float`, optional
            Upper y-limit for time series.

    **Outputs** :
        fig, axes : `matplotlib` figure and axes list
    """
    fig, axes = plt.subplots(3, 1, figsize=default_figsize, dpi=default_dpi, constrained_layout=True)

    # Time series with linear trend
    ax = axes[0]
    ax.plot(s.index, s, lw=0.8, color="black", alpha=0.8)

    if len(v) >= 2:
        xdays = (v.index - v.index[0]).days.values.astype(float)
        z = np.polyfit(xdays, v.values.astype(float), 1)
        slope_day = z[0]
        trend_per_year = slope_day * 365.25
        ax.plot(v.index, np.polyval(z, xdays), lw=1.0, ls="--", color="gray",
                label="Slope: %.2f m$^3$s$^{-1}$ yr$^{-1}$" % trend_per_year)

    years = np.arange(s.index.year.min(), s.index.year.max() + 1, 1)
    ax.set_xticks([pd.Timestamp(str(y)) for y in years])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y"))
    ax.set_xlim([s.index.min(), s.index.max()])
    if y_max is not None:
        ax.set_ylim([0, y_max])
    ax.set_ylabel("Discharge m$^3$s$^{-1}$")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_xlabel("Year")

    # Histogram with quantiles and optional threshold
    ax = axes[1]
    ax.hist(v, bins=50, color="0.6", alpha=0.7)
    for qlev, qv in q.items():
        ax.axvline(qv, color="0.2", ls="--", lw=0.9)
        ax.text(qv, ax.get_ylim()[1]*0.9, "q%d = %.0f" % (int(qlev*100), qv),
                rotation=90, va="top", ha="right", fontsize=7, color="0.2")
    if opening_threshold is not None:
        ax.axvline(opening_threshold, color="steelblue", lw=1.2)
        ax.text(opening_threshold, ax.get_ylim()[1]*0.9,
                "Opening threshold = %s" % (format(opening_threshold, ",.0f")),
                rotation=90, va="top", ha="right", fontsize=7, color="steelblue")
    ax.set_xlabel("Discharge m$^3$s$^{-1}$")
    ax.set_ylabel("Frequency")

    # Monthly violin plot
    ax = axes[2]
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    raw = [np.asarray(v[v.index.month == m].astype(float).values) for m in range(1, 13)]
    monthly = [x[np.isfinite(x)] for x in raw]
    positions = [i+1 for i, x in enumerate(monthly) if x.size > 0]
    data_for_plot = [x for x in monthly if x.size > 0]
    labels_for_plot = [lab for lab, x in zip(month_labels, monthly) if x.size > 0]

    vp = ax.violinplot(data_for_plot, positions=positions, widths=0.9,
                       showmeans=False, showextrema=False, showmedians=True)
    for body in vp.get('bodies', []):
        body.set_alpha(0.5)
        body.set_facecolor("0.5")
        body.set_edgecolor("0.2")
        body.set_linewidth(0.6)
    if 'cmedians' in vp and vp['cmedians'] is not None:
        vp['cmedians'].set_color("0.1")
        vp['cmedians'].set_linewidths(1.0)

    if opening_threshold is not None:
        ax.axhline(y=opening_threshold, linestyle="-", linewidth=1.0, color="steelblue")
        xmin, xmax = ax.get_xlim()
        ax.text(xmax - 0.5, opening_threshold,
                "Opening threshold (%s m³/s)" % format(opening_threshold, ",.0f"),
                ha="right", va="bottom", fontsize=7, color="steelblue")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_for_plot)
    ax.set_ylabel("Discharge m$^3$ s$^{-1}$")

    return fig, axes


# --------------------------------------------------------
# Model result figures (CSTM)
# --------------------------------------------------------
def plot_model_results(results, forecast_horizons):
    """Plot training/evaluation metrics across forecast horizons.

    **Inputs** :
        results : `dict`
            results[horizon][metric] -> (n_runs, n_epochs).
        forecast_horizons : `list`
            e.g., [1, 3, 5, 7]

    **Outputs** :
        fig, axs : `matplotlib` figure and axes array
    """
    metrics = ['train_loss', 'r2', 'willmott']
    titles = ['Training Loss (MSE)', 'R² Score', 'Willmott Index']
    ylabels = ['Loss (MSE)', r'Pearson correlation $(R^2)$', r'Willmott index ($d$)']
    colors, linestyles = _palette()

    fig, axs = plt.subplots(1, 3, figsize=default_figsize, sharex=True, sharey=True, dpi=default_dpi)

    for i, metric in enumerate(metrics):
        ax = axs[i]
        for j, horizon in enumerate(forecast_horizons):
            runs = np.array(results[horizon][metric])  # (n_runs, epochs)
            mean = runs.mean(axis=0)
            std = runs.std(axis=0)
            epochs = np.arange(mean.shape[0])

            c = colors[j % len(colors)]
            ls = linestyles[j % len(linestyles)]
            ax.plot(epochs, mean, label='%s-day' % horizon, color=c, linestyle=ls, linewidth=1.0)
            ax.fill_between(epochs, mean - std, mean + std, color=c, alpha=0.15)

        ax.set_ylabel(ylabels[i], fontsize=9)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        ax.set_title(titles[i], fontsize=9)

    for k in range(3):
        axs[k].set_ylim([-0.2, 1.0])
    axs[1].set_xlabel('Epoch', fontsize=10)

    axs[2].legend(title='Forecast', frameon=False, fontsize=8, title_fontsize=9,
                  handlelength=1.2, handletextpad=0.3, labelspacing=0.25,
                  loc='upper left', bbox_to_anchor=(1.02, 1.0))

    fig.tight_layout(pad=0.45, rect=[0, 0, 0.85, 1])
    return fig, axs


def plot_timeseries(preds_dict, forecast_horizons, key_true, key_pred, key_dates,
                    date_start, date_end, ylim=(0, 1_750_000)):
    """Plot measured vs. predicted discharge across horizons.

    **Inputs** :
        preds_dict : `dict`
            preds_dict[h][0] contains keys [key_true, key_pred, key_dates].
        forecast_horizons : `list`
            Horizons to plot.
        key_true, key_pred, key_dates : `str`
            Keys for observed, predicted, and datetime arrays.
        date_start, date_end : `str or datetime`
            Date window to plot.
        dpi : `int`
            Figure dpi.
        ylim : `tuple`
            y-limits.

    **Outputs** :
        fig, ax : `matplotlib` figure and axis
    """
    colors, linestyles = _palette()
    fig, ax = plt.subplots(figsize=default_figsize, dpi=default_dpi)

    for idx, horizon in enumerate(forecast_horizons):
        preds = preds_dict[horizon][0]
        y_true = np.asarray(preds[key_true]).ravel()
        y_pred = np.asarray(preds[key_pred]).ravel()
        dates = pd.to_datetime(preds[key_dates])

        mask = (dates >= pd.to_datetime(date_start)) & (dates <= pd.to_datetime(date_end))

        if idx == 0:
            ax.plot(dates[mask], y_true[mask], label="Measured",
                    color="black", linestyle="-", linewidth=1.0)

        ax.plot(dates[mask], y_pred[mask], label="%s Day" % horizon,
                color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)],
                alpha=0.8, linewidth=1.0)

    ax.set_ylabel("Discharge (m³/s)", fontsize=9)
    ax.legend(title='Forecast', frameon=False, fontsize=8, title_fontsize=9,
              handlelength=1.2, handletextpad=0.3, labelspacing=0.05,
              loc='upper left', bbox_to_anchor=(1.02, 1.0), ncols=1)

    ax.set_ylim(ylim)
    ax.set_xlim([pd.to_datetime(date_start) - pd.Timedelta(days=5),
                 pd.to_datetime(date_end) + pd.Timedelta(days=5)])
    ax.tick_params(labelsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.tight_layout()
    return fig, ax

