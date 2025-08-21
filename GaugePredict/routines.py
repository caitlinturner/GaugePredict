# routines.py  — simple, no type hints

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from dataretrieval import nwis

warnings.filterwarnings("ignore")


# figure defaults 
default_figsize = (7.33, 3.54)
default_dpi = 400
conus_extent = (-14_000_000, -7_300_000, 2_600_000, 7_000_000)
conus_ak_extent = (-20_000_000, -7_000_000, 2_600_000, 11_700_000)

# helpers
def figax(figsize=default_figsize, dpi=default_dpi):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    return fig, ax


def _read_3857(fp):
    gdf = gpd.read_file(fp)
    if gdf.crs is None:
        raise ValueError(f"{fp} has no crs")
    return gdf if gdf.crs.to_epsg() == 3857 else gdf.to_crs(3857)


def _set_extent(ax, extent):
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def add_gray_basemap(ax, zoom=4):
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas,
                    attribution=False, zoom=zoom, crs=3857)


def _union_geom(gdf):
    return gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union



# Data loading by HUC 
def _discover_huc_codes(base_dir):
    hits = list(Path(base_dir).glob("HUC??/WBDHU2.shp"))
    if not hits:
        raise FileNotFoundError(f"no huc shapefiles under {base_dir}")
    return sorted({p.parent.name[-2:] for p in hits})


def load_hucs_3857(base_dir):
    codes = _discover_huc_codes(base_dir)
    gdfs = []
    for code in codes:
        shp = Path(base_dir) / f"HUC{code}" / "WBDHU2.shp"
        if not shp.exists():
            warnings.warn(f"missing: {shp} (skipping)")
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
        raise FileNotFoundError("no usable huc shapefiles after scanning")
    out = pd.concat(gdfs, ignore_index=True)
    return out if out.crs and out.crs.to_epsg() == 3857 else out.to_crs(3857)


def simple_huc_colors(codes):
    base = list(plt.colormaps["tab20"].colors)
    k = len(codes)
    palette = (base * int(np.ceil(k / len(base))))[:k]
    return {c: palette[i] for i, c in enumerate(sorted(codes))}



# plotting

def plot_hucs(
    base_dir,
    states_fp,
    include_ak=False,
    figsize=default_figsize,
    dpi=default_dpi,
    label_hucs=True,
    basemap=True,
    zoom=4,
):
    """plot huc polygons with state boundaries; CONUS by default."""
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

    fig, ax = figax(figsize=figsize, dpi=dpi)
    basins_plot.plot(ax=ax, facecolor=basins_plot["_color"],
                     edgecolor="dimgray", linewidth=0.4, alpha=0.3, zorder=1)
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



# gauge tools
# units: customary -> (cfs, ft, in); metric -> (m^3/s, m, cm)
_unit_converters = {
    "discharge": {
        ("customary", "metric"):  lambda s: s * 0.0283168466,   # cfs to m³/s
        ("metric",    "customary"): lambda s: s / 0.0283168466,
    },
    "water_level": {
        ("customary", "metric"):  lambda s: s * 0.3048,         # ft to m
        ("metric",    "customary"): lambda s: s / 0.3048,
    },
    "precipitation": {
        ("customary", "metric"):  lambda s: s * 2.54,           # in to cm
        ("metric",    "customary"): lambda s: s / 2.54,
    },
}

_param_defaults = {
    # daily values codes commonly used in NWIS
    "discharge":     "00060",  # cfs
    "water_level":   "00065",  # ft (gage height)
    "precipitation": "00045",  # inches, daily total
}


def build_full_index(start_date, end_date, tz="UTC"):
    idx = pd.date_range(start=start_date, end=end_date, freq="D")
    if getattr(idx, "tz", None) is None:
        return idx.tz_localize(tz)
    return idx.tz_convert(tz)


def _to_series(ts):
    s = pd.Series(ts) if not isinstance(ts, pd.Series) else ts.copy()
    s.index = pd.to_datetime(s.index)
    s = pd.to_numeric(s, errors="coerce")
    return s


def _convert_units(s, parameter, from_units, to_units):
    parameter = str(parameter).lower()
    key = (from_units.lower(), to_units.lower())
    if from_units == to_units:
        return s
    if parameter not in _unit_converters or key not in _unit_converters[parameter]:
        raise ValueError(f"unsupported conversion: parameter={parameter}, {from_units} to {to_units}")
    return _unit_converters[parameter][key](s)


def process_series(
    ts,
    index,
    parameter,                 # 'discharge' | 'water_level' | 'precipitation'
    to_units="metric",         # 'metric' | 'customary'
    assume_input_units="customary",
    force_tz="UTC",
    sentinels=(-999999, -99999, -9999),
    fill=True,
):
    """
    parse -> timezone normalize -> align to index -> unit convert -> optional fill
    """
    s = _to_series(ts).replace(list(sentinels), np.nan)

    if s.index.tz is None:
        s.index = s.index.tz_localize(force_tz)
    else:
        s.index = s.index.tz_convert(force_tz)
    idx = index if getattr(index, "tz", None) is not None else index.tz_localize(force_tz)

    s = s.sort_index().reindex(idx)
    s = _convert_units(s, parameter=parameter, from_units=assume_input_units, to_units=to_units)

    if fill:
        s = s.interpolate(limit_direction="both").ffill().bfill()
    return s


def load_site_json(json_path):
    with Path(json_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_site_series_from_json(
    raw,
    full_index,
    parameter,                     # 'discharge' | 'water_level' | 'precipitation'
    to_units="metric",
    assume_input_units="customary",
    series_key="parameter",
    alt_keys=("discharge", "water_level", "stage", "precipitation"),
):
    """
    Build dataset
    """
    out = {}
    for huc_sites in raw.values():
        for site_no, info in huc_sites.items():
            found = None
            if series_key in info:
                found = info[series_key]
            else:
                for k in alt_keys:
                    if k in info:
                        found = info[k]
                        break
            if found is None:
                continue
            s = process_series(
                found, full_index,
                parameter=parameter,
                to_units=to_units,
                assume_input_units=assume_input_units)
            out[site_no] = s
    return out


def _pick_dv_column(df, parameter_code, parameter):
    cols = [c for c in df.columns if parameter_code in c]
    if not cols:
        return None
    if parameter == "precipitation":
        prefer = ("sum", "total", "accum")
    else:
        prefer = ("mean", "value")
    def rank(name):
        name_l = name.lower()
        for i, tag in enumerate(prefer):
            if tag in name_l:
                return i
        return len(prefer)
    cols_sorted = sorted(cols, key=rank)
    return cols_sorted[0]


def load_target_series(
    target_site,
    start_date,
    end_date,
    parameter,                    # 'discharge' | 'water_level' | 'precipitation'
    parameter_code=None,
    to_units="metric",
    assume_input_units="customary",
    force_tz="UTC",
):
    """
    Download data
    """
    parameter = parameter.lower()
    code = parameter_code or _param_defaults.get(parameter)
    if code is None:
        raise ValueError(f"no default parameter code for parameter='{parameter}'")

    idx = build_full_index(start_date, end_date, tz=force_tz)
    dv = nwis.get_dv(sites=target_site, parameterCd=code, start=idx[0], end=idx[-1])[0]

    col = _pick_dv_column(dv, code, parameter)
    if col is None:
        raise RuntimeError(f"no dv column found for parameter {code} at site {target_site}")

    s = dv[col]
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize(force_tz)
    else:
        s.index = s.index.tz_convert(force_tz)

    s = s.sort_index().reindex(idx).interpolate(limit_direction="both").ffill().bfill()
    s = _convert_units(s, parameter=parameter, from_units=assume_input_units, to_units=to_units)
    return s


def stack_features(
    site_series_by_site,
    target_series,
    include_target=True,
    include_target_diff=True,
):
    """
    build feature tensor and target arrays.
    returns (x_raw [n_channels, n_times], y [n_times], feature_names).
    """
    feats = []
    names = []

    for site_no, s in site_series_by_site.items():
        feats.append(s.reindex_like(target_series))
        names.append(f"site_{site_no}")

    if include_target:
        feats.append(target_series)
        names.append("target")

    if include_target_diff:
        feats.append(target_series.diff().bfill())
        names.append("target_diff")

    x_raw = np.stack([s.values for s in feats], axis=0)
    y = target_series.values.copy()
    return x_raw, y, names




import matplotlib.dates as mdates

def compute_discharge_stats(
    target: pd.Series,
    start_date: str,
    end_date: str,
    *,
    tz: str | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Discharge data figures and stats.

    Returns:
    stats : DataFrame (one row of summary stats)
    s     : Series, daily reindexed (no interpolation)
    v     : Series, valid values (dropna)
    q     : Series, quantiles (0.05, 0.25, 0.50, 0.75, 0.95)
    """
    # timezone handling
    tz_out = tz or getattr(getattr(target.index, "tz", None), "key", "UTC") or "UTC"

    # daily index and raw reindex (no filling here)
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


def plot_discharge_panels(
    s: pd.Series,
    v: pd.Series,
    q: pd.Series,
    *,
    opening_threshold: float | None = None,
    y_max: float | None = None,
    figsize=(7, 7),
    dpi=300
):
    """
    time series w/ trend, histogram w/ quantiles, monthly violins.
    Threshold line is optional.
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=dpi, constrained_layout=True)

    # Time Series
    ax = axes[0]
    ax.plot(s.index, s, color="gray", lw=0.8)

    if len(v) >= 2:
        xdays = (v.index - v.index[0]).days.values.astype(float)
        z = np.polyfit(xdays, v.values.astype(float), 1)
        slope_day = z[0]
        trend_per_year = slope_day * 365.25
        ax.plot(v.index, np.polyval(z, xdays), color="steelblue", lw=1, ls="--",
                label=f"Slope: {trend_per_year:.2f} m$^3$s$^{-1}$ yr$^{-1}$")

    years = np.arange(s.index.year.min(), s.index.year.max() + 1, 1)
    ax.set_xticks([pd.Timestamp(str(y)) for y in years])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y"))
    ax.set_xlim([s.index.min(), s.index.max()])
    if y_max is not None:
        ax.set_ylim([0, y_max])
    ax.set_ylabel("Discharge m$^3$s$^{-1}$")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_xlabel("Year")

    # Histogram with quantiles 
    ax = axes[1]
    ax.hist(v, bins=50, color="gray", alpha=0.7)

    for qlev, qv in q.items():
        ax.axvline(qv, color="black", ls="--")
        ax.text(qv, ax.get_ylim()[1]*0.9, f"q{int(qlev*100)} = {qv:.0f}",
                rotation=90, va="top", ha="right", color="black", fontsize=7)

    if opening_threshold is not None:
        ax.axvline(opening_threshold, color="steelblue", ls="-", lw=1.5)
        ax.text(opening_threshold, ax.get_ylim()[1]*0.9,
                f"Opening threshold = {opening_threshold:,.0f}",
                rotation=90, va="top", ha="right", color="steelblue", fontsize=7)

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

    vp = ax.violinplot(
        data_for_plot,
        positions=positions,
        widths=0.9,
        showmeans=False,
        showextrema=False,
        showmedians=True
    )
    for body in vp['bodies']:
        body.set_alpha(0.5)
        body.set_facecolor("darkgray")
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
    if 'cmedians' in vp and vp['cmedians'] is not None:
        vp['cmedians'].set_color("black")
        vp['cmedians'].set_linewidths(1.0)

    if opening_threshold is not None:
        ax.axhline(y=opening_threshold, linestyle="-", linewidth=1.2, color="steelblue")
        xmin, xmax = ax.get_xlim()
        ax.text(xmax - 0.5, opening_threshold,
                f"Opening threshold ({opening_threshold:,.0f} m³/s)",
                ha="right", va="bottom", fontsize=7, color="steelblue")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_for_plot)
    ax.set_ylabel("Discharge m$^3$ s$^{-1}$")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    return fig, axes
