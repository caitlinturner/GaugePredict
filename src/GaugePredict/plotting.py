# -*- coding: utf-8 -*-
"""
GaugePredict/plotting.py

Plotting utilities for model outputs, SHAP summaries, and geospatial context.

"""

from __future__ import division, print_function, absolute_import

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

import contextily as ctx
import cmocean

from .routines import load_hucs_3857


# =============================================================================
# Utils
# =============================================================================

def get_examples_results_dir(project_root):
    """
        Return the default examples/results directory under a project root.
    
        **Inputs** :
    
            project_root : 'str or pathlib.Path'
                Project root directory.
    
        **Outputs** :
    
            results_dir : 'pathlib.Path'
                Path to "<project_root>/examples/results".
    """
    return Path(project_root) / "examples" / "results"


def parameter_label_from_target(target_variable):
        """
            Create a standard y-axis label from a target-variable name.
        
            Currently supports a discharge label with 10^4 scaling and a default
            water-level label. Broad use will be updated.
        
            **Inputs** :
        
                target_variable : 'str'
                    Target variable identifier (e.g., "discharge", "water_level").
        
            **Outputs** :
        
                label : 'str'
                    Matplotlib-ready label string.
        """
    if str(target_variable).lower() == "discharge":
        return r"Discharge ($10^{4}$ $m^{3}$ $s^{-1}$)"
    return r"Water level (m)"


def horizon_dir(results_root, h):
    """
        Construct a standardized subdirectory path for a forecast horizon.
    
        Example:
            horizon_dir("results", 3) -> Path("results") / "H03"
    
        **Inputs** :
    
            results_root : 'str or pathlib.Path'
                Root directory containing horizon folders.
    
            h : 'int'
                Forecast horizon.
    
        **Outputs** :
    
            path : 'pathlib.Path'
                Horizon directory path.
    """
    return Path(results_root) / f"H{int(h):02d}"

def build_aligned_test_series(results, horizons):
    """
    Build an aligned observed series and a long-form prediction table across horizons.
    This function intersects the available test date ranges across all horizons so
    that predictions and observations are aligned on a common set of dates. Observations 
    are taken from the largest horizon (max(horizons_sorted)) run
    to provide a consistent y_true vector aligned to the intersection index.

    **Inputs** :

        results : 'dict'
            Mapping horizon -> loaded run dict (see load_saved_horizon_run()).

        horizons : 'iterable'
            Horizons to include (only those present in results are used).

    **Outputs** :

        date_index : 'pandas.DatetimeIndex'
            Intersection of all test date indices across included horizons.

        y_true : 'numpy.ndarray'
            Observed values aligned to date_index.

        pred_df : 'pandas.DataFrame'
            Long-form table with columns ["date", "horizon", "y_pred"] aligned to date_index.
    """
    horizons_sorted = sorted(int(h) for h in horizons if int(h) in results)

    date_index = None
    for h in horizons_sorted:
        d = pd.to_datetime(results[h]["dates_test"])
        date_index = d if date_index is None else date_index.intersection(d)

    base_h = max(horizons_sorted)
    dates_base = pd.to_datetime(results[base_h]["dates_test"])
    y_true_base = np.asarray(results[base_h]["y_true_test"], dtype=float)
    y_true = pd.Series(y_true_base, index=dates_base).reindex(date_index).to_numpy()

    pred_rows = []
    for h in horizons_sorted:
        d = pd.to_datetime(results[h]["dates_test"])
        y_pred = np.asarray(results[h]["y_pred_test"], dtype=float)
        s = pd.Series(y_pred, index=d).reindex(date_index)
        pred_rows.append(pd.DataFrame({"date": date_index, "horizon": int(h), "y_pred": s.to_numpy()}))

    pred_df = pd.concat(pred_rows, ignore_index=True)
    return date_index, y_true, pred_df


def get_horizon_styles(horizons, cmap=None, min_color=0.15, max_color=0.9):
    """
        Assign a distinct color and linestyle for each horizon.
        Colors are sampled from a continuous colormap over [min_color, max_color].
        Linestyles cycle through a predefined set.
    
        **Inputs** :
    
            horizons : 'iterable'
                Horizons to style.
    
            cmap : 'matplotlib colormap or None'
                Colormap used for horizon colors. Defaults to cmocean.cm.haline.
    
            min_color, max_color : 'float'
                Fractions in [0, 1] used for colormap sampling range.
    
        **Outputs** :
    
            colors_h : 'dict'
                Mapping horizon -> RGBA color.
    
            linestyles_h : 'dict'
                Mapping horizon -> matplotlib linestyle spec.
    """
    cmap = cmap or cmocean.cm.haline
    horizons_sorted = sorted(int(h) for h in horizons)
    den = max(1, len(horizons_sorted) - 1)

    linestyles_cycle = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

    colors_h = {}
    linestyles_h = {}
    for idx, h in enumerate(horizons_sorted):
        frac = min_color + (max_color - min_color) * (idx / den)
        colors_h[h] = cmap(frac)
        linestyles_h[h] = linestyles_cycle[idx % len(linestyles_cycle)]
    return colors_h, linestyles_h


def build_scores_table(results, horizons):
    """
    Build a summary table of evaluation metrics by horizon.

    **Inputs** :

        results : 'dict'
            Mapping horizon -> loaded run dict.

        horizons : 'iterable'
            Horizons to include (only those present in results are used).

    **Outputs** :

        df : 'pandas.DataFrame'
            DataFrame indexed by horizon with columns ["r2","nse","willmott"].
    """
    rows = []
    for h in sorted(int(h) for h in horizons if int(h) in results):
        m = results[h]["metr"]
        rows.append({"horizon": int(h), "r2": float(m["r2"]), "nse": float(m["nse"]), "willmott": float(m["willmott"])})
    if not rows:
        return pd.DataFrame(columns=["r2", "nse", "willmott"])
    return pd.DataFrame(rows).set_index("horizon").sort_index()


# =============================================================================
# Load saved model outputs
# =============================================================================

def load_saved_horizon_run(results_root, h, *, verbose=True):
    """
        Load saved model outputs for a single forecast horizon.
    
        Expects the horizon directory to contain:
          - predictions.csv (date, y_true, y_pred)
          - metrics.json
          - history.json
          - model.pt
          - scaler_y.pkl
    
        Dates in predictions.csv are parsed as UTC and then converted to naive
        timestamps for plotting convenience.
    
        **Inputs** :
    
            results_root : 'str or pathlib.Path'
                Root directory containing horizon subfolders.
    
            h : 'int'
                Forecast horizon to load.
    
            verbose : 'bool'
                If True, prints a message when required files are missing.
    
        **Outputs** :
    
            run : 'dict or None'
                Dictionary containing:
                  - dates_test : numpy array of datetime-like (tz-naive)
                  - y_true_test, y_pred_test : numpy arrays
                  - metr : dict of metrics
                  - history : dict of training curves
                  - scaler_y : loaded scaler object
                  - model_path : pathlib.Path to model.pt
                Returns None if required files are missing.
    """
    d = horizon_dir(results_root, h)

    req = [
        d / "predictions.csv",
        d / "metrics.json",
        d / "history.json",
        d / "model.pt",
        d / "scaler_y.pkl",
    ]

    if not all(p.exists() for p in req):
        if verbose:
            missing = [str(p.name) for p in req if not p.exists()]
            print(f"[H={int(h):02d}] missing {missing} in {d}")
        return None

    df_pred = pd.read_csv(d / "predictions.csv")
    dates = pd.to_datetime(df_pred["date"], utc=True).dt.tz_convert(None)

    y_true = df_pred["y_true"].astype(float).to_numpy()
    y_pred = df_pred["y_pred"].astype(float).to_numpy()

    with open(d / "metrics.json", "r", encoding="utf-8") as f:
        metr = json.load(f)

    with open(d / "history.json", "r", encoding="utf-8") as f:
        hist = json.load(f)

    with open(d / "scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)

    return {
        "dates_test": dates.to_numpy(),
        "y_true_test": y_true,
        "y_pred_test": y_pred,
        "metr": metr,
        "history": hist,
        "scaler_y": scaler_y,
        "model_path": d / "model.pt",
    }


def load_saved_runs(results_root, horizons, *, verbose=True, require_any=True):
    """
        Load saved runs for multiple horizons.
    
        **Inputs** :
    
            results_root : 'str or pathlib.Path'
                Root directory containing per-horizon subfolders.
    
            horizons : 'iterable'
                Horizons to load.
    
            verbose : 'bool'
                If True, prints a message per loaded horizon and for missing horizons.
    
            require_any : 'bool'
                If True, raises if no horizons are found.
    
        **Outputs** :
    
            results : 'dict'
                Mapping horizon (int) -> run dict from load_saved_horizon_run().
    
        **Raises** :
    
            RuntimeError
                If require_any is True and no runs are found.
    """
    results = {}
    for h in horizons:
        run = load_saved_horizon_run(results_root, h, verbose=verbose)
        if run is not None:
            results[int(h)] = run
            if verbose:
                print(f"Loaded H={int(h)}")

    if require_any and not results:
        raise RuntimeError(f"No runs found in {results_root}")

    return results


# =============================================================================
# SHAP utils
# =============================================================================

def load_shap_tables_by_horizon(shap_root, horizons, *, filename="shap_sites.csv", verbose=True):
    """
        Load SHAP site-importance CSV files across multiple horizons and concatenate.
    
        Each file is expected at:
            <shap_root>/H##/<filename>
    
        The output includes an added "horizon" column.
    
        **Inputs** :
    
            shap_root : 'str or pathlib.Path'
                Root directory containing horizon subfolders.
    
            horizons : 'iterable'
                Horizons to attempt to load.
    
            filename : 'str'
                CSV filename to load from each horizon folder.
    
            verbose : 'bool'
                If True, prints a message when a horizon file is missing.
    
        **Outputs** :
    
            df : 'pandas.DataFrame'
                Concatenated SHAP table with "horizon" column.
    
        **Raises** :
    
            RuntimeError
                If no SHAP files are found for the requested horizons.
    """
    shap_root = Path(shap_root)
    frames = []

    for h in horizons:
        shap_csv = shap_root / f"H{int(h):02d}" / filename
        if not shap_csv.exists():
            if verbose:
                print(f"[H={int(h)}] missing {filename} at {shap_csv}, skipping")
            continue
        df = pd.read_csv(shap_csv)
        df["horizon"] = int(h)
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No SHAP files found under {shap_root} for {filename}")

    return pd.concat(frames, ignore_index=True)


def load_states(states_fp):
    """
        Load a states boundary file and standardize to EPSG:4326.
    
        If the input file has no CRS, EPSG:4269 is assumed (common for some US datasets).
        If available, AK, HI, and US territories are removed to focus on CONUS.
    
        **Inputs** :
    
            states_fp : 'str or pathlib.Path'
                Path to a states shapefile/GeoPackage/GeoJSON supported by geopandas.
    
        **Outputs** :
    
            states : 'geopandas.GeoDataFrame'
                States boundaries projected to EPSG:4326, filtered to CONUS when possible.
    """
    states = gpd.read_file(states_fp)

    if states.crs is None:
        states = states.set_crs(4269)

    states = states.to_crs(4326)

    if "STUSPS" in states.columns:
        states = states[~states["STUSPS"].isin(["AK", "HI", "PR", "VI"])].copy()

    return states


def _normalize_importance_to_unit(df, *, imp_col="importance", out_col="importance_norm"):
    """
        Ensure a SHAP importance column is normalized to [0, 1].
    
        Behavior:
        - If out_col already exists, it is clipped to [0, 1] after NaN/inf handling.
        - Otherwise, imp_col is normalized by its maximum (after clipping negatives to 0).
    
        **Inputs** :
    
            df : 'pandas.DataFrame'
                SHAP table.
    
            imp_col : 'str'
                Column containing raw importance values.
    
            out_col : 'str'
                Output column name for normalized importance.
    
        **Outputs** :
    
            df_out : 'pandas.DataFrame'
                Copy of df with a valid out_col in [0, 1].
    
        **Raises** :
    
            ValueError
                If neither out_col nor imp_col exists.
    """
    d = df.copy()

    if out_col in d.columns:
        vals = d[out_col].to_numpy(dtype=float)
        d[out_col] = np.clip(np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        return d

    if imp_col not in d.columns:
        raise ValueError(f"Missing '{out_col}' or '{imp_col}' in SHAP table.")

    vals = d[imp_col].to_numpy(dtype=float)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.clip(vals, 0.0, None)

    vmax = float(np.nanmax(vals)) if vals.size else 0.0
    d[out_col] = 0.0 if vmax <= 0.0 else (vals / vmax)

    d[out_col] = np.clip(
        np.nan_to_num(d[out_col].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0,
    )
    return d


def load_shap_sites_csv(shap_root, h):
    """
        Load and normalize a horizon-specific SHAP site-importance table.
    
        Expects:
            <shap_root>/H??/shap_sites.csv
    
        Required columns: "lat", "lon"
        Normalized importance is returned in "importance_norm".
    
        **Inputs** :
    
            shap_root : 'str or pathlib.Path'
                Root directory containing SHAP outputs.
    
            h : 'int'
                Horizon to load.
    
        **Outputs** :
    
            df : 'pandas.DataFrame'
                SHAP table with importance normalized to [0, 1].
    
        **Raises** :
    
            FileNotFoundError
                If shap_sites.csv does not exist.
    
            ValueError
                If required columns are missing.
    """
    csv_path = Path(shap_root) / f"H{int(h):02d}" / "shap_sites.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"SHAP file not found for H={int(h)}: {csv_path}")

    df = pd.read_csv(csv_path)

    for c in ["lat", "lon"]:
        if c not in df.columns:
            raise ValueError(f"{c} column missing in {csv_path}")

    return _normalize_importance_to_unit(df, imp_col="importance", out_col="importance_norm")


def top_n_shap_sites(df, n_keep):
    """
        Select the top-N sites by normalized SHAP importance.
    
        **Inputs** :
    
            df : 'pandas.DataFrame'
                SHAP site table containing "importance_norm".
    
            n_keep : 'int'
                Number of rows to keep.
    
        **Outputs** :
    
            df_top : 'pandas.DataFrame'
                Copy of top-N sites sorted by descending importance.
    """
    return df.sort_values("importance_norm", ascending=False).head(int(n_keep)).copy()


# =============================================================================
# SHAP geoplot grid 
# =============================================================================

def plot_shap_geoplot_grid(
    *,
    shap_root,
    horizons,
    n_shap_by_h,
    states_fp=None,
    xlim=None,
    ylim=None,
    fig_w=8.0,
    fig_h=6.0,
    nrows=None,
    ncols=None,
    s_all=6.0,
    s_used=18.0,
    wspace=0.03,
    hspace=-0.0125,
    cbar_rect=(0.125, 0.07, 0.775, 0.03),
    save_path=None,
    show=True,
    dpi=300,
    save_dpi=400,
    font_size=8,
):
    """
        Creates subplot of of SHAP site importance maps for multiple horizons.
    
        Each figure shows:
        - All available sites in light gray
        - The top-N sites (per horizon) colored by normalized SHAP importance
    
        State boundaries are optional. If states_fp is provided, boundaries are drawn
        for geographic context; if states_fp is None, boundaries are skipped.

        This was specific for project use. A generalized update to this function is comming soon.
    
        **Inputs** :
    
            shap_root : 'str or pathlib.Path'
                Root directory containing per-horizon SHAP outputs.
    
            horizons : 'iterable'
                Horizons to plot.
    
            n_shap_by_h : 'dict'
                Mapping horizon -> number of top sites to highlight.
    
            states_fp : 'str or pathlib.Path'
                States boundary dataset path (read by geopandas).
    
            xlim, ylim : 'tuple (float, float)' or None
                Plot bounds in degrees (lon/lat) for EPSG:4326 output.
    
            fig_w, fig_h : 'float'
                Figure width/height in inches.
    
            nrows, ncols : 'int'  or None
                Grid arrangement for panels.
    
            s_all : 'float'
                Marker size for all sites.
    
            s_used : 'float'
                Marker size for highlighted sites.
    
            wspace, hspace : 'float'
                Grid spacing.
    
            cbar_rect : 'tuple'
                Rectangle (left, bottom, width, height) for colorbar axes in figure
                fraction coordinates.
    
            save_path : 'str or pathlib.Path or None'
                If provided, figure is saved to this path.
    
            show : 'bool'
                If True, calls plt.show().
    
            dpi : 'int'
                Figure display dpi.
    
            save_dpi : 'int'
                Save dpi when writing to disk.
    
            font_size : 'int'
                Base matplotlib font size.
    
        **Outputs** :
    
            fig : 'matplotlib.figure.Figure'
                The created figure.
    
        **Raises** :
    
            KeyError
                If n_shap_by_h is missing an entry for one of the requested horizons.
    """
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": save_dpi,
        "font.size": font_size,
        "axes.linewidth": 0.8,
    })

    states_gdf = load_states(states_fp)

    frames = {}
    used_frames = {}

    for h in horizons:
        h = int(h)
        df_h = load_shap_sites_csv(shap_root, h)
        frames[h] = df_h

        if h not in n_shap_by_h:
            raise KeyError(f"n_shap_by_h missing entry for horizon H={h}")

        used_frames[h] = top_n_shap_sites(df_h, n_shap_by_h[h])

    fig = plt.figure(figsize=(fig_w, fig_h))
    grid = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=wspace, hspace=hspace)

    cmap = cmocean.cm.haline
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    scatter_used_last = None

    for j, h in enumerate(horizons):
        h = int(h)
        row = j // ncols
        col = j % ncols
        ax = fig.add_subplot(grid[row, col])

        df_all = frames[h]
        df_used = used_frames[h]

        ax.scatter(
            df_all["lon"].to_numpy(),
            df_all["lat"].to_numpy(),
            s=s_all,
            color="0.85",
            edgecolor="none",
            zorder=1,
        )

        v = np.clip(df_used["importance_norm"].to_numpy(dtype=float), 0.0, 1.0)

        scatter_used = ax.scatter(
            df_used["lon"].to_numpy(),
            df_used["lat"].to_numpy(),
            s=s_used,
            c=v,
            cmap=cmap,
            norm=norm,
            edgecolor="k",
            linewidth=0.25,
            alpha=0.95,
            zorder=3,
        )
        scatter_used_last = scatter_used

        states_gdf.boundary.plot(
            ax=ax,
            color="0.6",
            linewidth=0.35,
            zorder=0,
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)

        n_used = int(df_used.shape[0])
        ax.text(
            0.98,
            0.95,
            f"{h} days\n({n_used} sites)",
            transform=ax.transAxes,
            fontsize=7,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", alpha=0.75, pad=1.2, edgecolor="none"),
            zorder=10,
        )

        for loc in ["left", "right", "top", "bottom"]:
            ax.spines[loc].set_linewidth(0.9)

    if scatter_used_last is not None:
        cbar_ax = fig.add_axes(list(cbar_rect))
        cbar = fig.colorbar(scatter_used_last, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("SHAP importance (normalized)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax

# =============================================================================
# HUC plotting
# =============================================================================

conus_extent = (-14_000_000, -7_300_000, 2_600_000, 7_000_000)
conus_ak_extent = (-20_000_000, -7_000_000, 2_600_000, 11_700_000)

def plot_hucs(
    base_dir,
    states_fp,
    *,
    include_ak=False,
    label_hucs=True,
    basemap=True,
    zoom=4,
):
    """
    Plot basin polygons with optional state boundaries and basemap.

    By default, this function produces a CONUS-focused plot:
    - Alaska, Hawaii, and territories are excluded in default
    - HUC2 code "19" (Alaska) is excluded from basins

    If include_ak=True, the plot is produced in EPSG:3857 and Alaska is included.

    **Inputs** :

        base_dir : 'str or pathlib.Path'
            Root directory containing HUC??/WBDHU2.shp shapefiles.

        states_fp : 'str or pathlib.Path'
            States boundary dataset path.

        include_ak : 'bool'
            If True, include Alaska and use EPSG:3857. If False, use EPSG:4326.

        label_hucs : 'bool'
            If True, annotate each HUC2 polygon group with its HUC code.

        basemap : 'bool'
            If True, add a contextily basemap (requires internet tile access).

        zoom : 'int'
            Contextily basemap zoom level.

        xlim, ylim : 'tuple or None'
            Optional axis limits in the current CRS units. If None, defaults are used.

    **Outputs** :

        fig : 'matplotlib.figure.Figure'
            Figure object.

        ax : 'matplotlib.axes.Axes'
            Axes object.
    """
    basins = load_hucs_3857(base_dir)

    states = gpd.read_file(states_fp)
    if states.crs is None:
        raise ValueError(f"{states_fp} has no CRS")
    states = states.rename(columns=str.lower)

    target_epsg = 4326 if not include_ak else 3857

    if basins.crs is None or basins.crs.to_epsg() != target_epsg:
        basins = basins.to_crs(target_epsg)
    if states.crs.to_epsg() != target_epsg:
        states = states.to_crs(target_epsg)

    if include_ak:
        states_plot = states
        basins_plot = basins
    else:
        states_plot = states[~states["stusps"].isin(["AK", "HI", "PR", "VI"])]
        basins_plot = basins[basins["huc2"] != "19"]

    unique_codes = sorted(basins_plot["huc2"].unique())
    base_palette = list(plt.colormaps["tab20"].colors)
    k = len(unique_codes)
    palette = (base_palette * int(np.ceil(k / len(base_palette))))[:k]
    color_map = {c: palette[i] for i, c in enumerate(unique_codes)}
    basins_plot = basins_plot.assign(_color=basins_plot["huc2"].map(color_map))

    fig, ax = plt.subplots(figsize=(7.33, 3.54), dpi=300, constrained_layout=True)

    if target_epsg == 4326:
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
    else:
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

    basins_plot.plot(
        ax=ax,
        facecolor=basins_plot["_color"],
        edgecolor="dimgray",
        linewidth=0.4,
        alpha=0.30,
        zorder=1,
    )
    states_plot.boundary.plot(
        ax=ax,
        linewidth=0.5,
        edgecolor="gray",
        alpha=0.6,
        zorder=2,
    )

    if label_hucs:
        try:
            for h, sub in basins_plot.groupby("huc2"):
                rp = sub.geometry.union_all().representative_point()
                ax.text(rp.x, rp.y, h, ha="center", va="center", fontsize=8, fontweight="bold", zorder=3)
        except Exception:
            for h, sub in basins_plot.groupby("huc2"):
                rp = sub.unary_union.representative_point()
                ax.text(rp.x, rp.y, h, ha="center", va="center", fontsize=8, fontweight="bold", zorder=3)

    if basemap:
        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldGrayCanvas,
            attribution=False,
            zoom=zoom,
            crs=target_epsg,
        )

    xmin, ymin, xmax, ymax = states_plot.total_bounds
    pad_x = (xmax - xmin) * 0.02
    pad_y = (ymax - ymin) * 0.02
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    ax.grid(False)
    return fig, ax

# =============================================================================
# Target Site plotting
# =============================================================================
def plot_statistics(
    target,
    *,
    critical_threshold=None,
    figsize=(6.00, 5.4),
    dpi=400,
    ylim=None,
    hist_bins=40,
    show_trend=True,
    trend_label=None,
):
    """
    Subplot of statistics for a daily time series at a target site.

    Panels:
      (a) time series and optional linear trend
      (b) histogram with quantile markers and optional critical threshold
      (c) monthly violin plots and optional critical threshold

    **Inputs** :

        target : 'pandas.Series'
            Daily series indexed by datetimes. Timezone-aware (UTC) is recommended.

        critical_threshold : 'float or None'
            Threshold value to annotate on histogram and violin plot.

        figsize : 'tuple (float, float)'
            Figure size in inches.

        dpi : 'int'
            Figure DPI.

        ylim : 'tuple (float, float) or None'
            (ymin, ymax) limits for panels (a) and (c). If None, auto-scaled.

        hist_bins : 'int'
            Number of bins for histogram.

        show_trend : 'bool'
            If True, fit and plot a linear trend on panel (a).

        trend_label : 'str or None'
            If provided, overrides the trend legend label.

    **Outputs** :

        fig : 'matplotlib.figure.Figure'
            Figure object.

        axes : 'numpy.ndarray of matplotlib.axes.Axes'
            Array of axes for panels (a), (b), and (c).

    **Raises** :

        ValueError
            If target is None or contains fewer than 2 finite values.

        TypeError
            If target is not a pandas Series-like object.
    """


    if target is None:
        raise ValueError("target is None")
    if not hasattr(target, "index"):
        raise TypeError("target must be a pandas Series")

    s = pd.Series(target.copy())
    s = s.sort_index()

    s.index = pd.to_datetime(s.index)

    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")

    v = s.dropna()
    if v.size < 2:
        raise ValueError("target must have at least 2 finite values")

    q = v.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    haline_color = cmocean.cm.haline(0.3)

    fig, axes = plt.subplots(
        3, 1,
        figsize=figsize,
        dpi=dpi,
        constrained_layout=True,
    )

    # panel (a): time series 
    ax1 = axes[0]
    ax1.plot(s.index, s.to_numpy(dtype=float), lw=0.8, color="black", alpha=0.8, label="Observed")

    if show_trend:
        vv = v.astype(float)
        xdays = (vv.index - vv.index[0]).days.astype(float)
        z = np.polyfit(xdays, vv.to_numpy(), 1)
        y_trend = np.polyval(z, xdays)

        slope_per_day = float(z[0])
        slope_per_yr = slope_per_day * 365.25

        if trend_label is None:
            trend_label_use = f"Slope: {slope_per_yr:,.2f} m³ s⁻¹ yr⁻¹"
        else:
            trend_label_use = str(trend_label)

        ax1.plot(vv.index, y_trend, lw=1.0, ls="--", color=haline_color, label=trend_label_use)

    years = np.arange(s.index.year.min(), s.index.year.max() + 1)
    label_years = years[years % 2 == 0]
    ax1.set_xticks([pd.Timestamp(str(y)).tz_localize(s.index.tz) for y in label_years])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax1.set_xlim(s.index.min(), s.index.max())
    ax1.set_ylabel("Discharge (m³ s⁻¹)", fontsize=7, fontweight="bold")
    if ylim is not None:
        ax1.set_ylim(*ylim)
    ax1.tick_params(labelsize=7)
    ax1.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    ax1.grid(False)
    _despine_axes(ax1)

    # panel (b): histogram 
    ax2 = axes[1]
    ax2.hist(v.to_numpy(dtype=float), bins=int(hist_bins), color="0.8")
    ax2.set_ylabel("Frequency", fontsize=7, fontweight="bold", labelpad=13.5)
    ax2.tick_params(labelsize=7)

    y_top = ax2.get_ylim()[1]
    text_y_top = 0.95 * y_top
    xmin, xmax = ax2.get_xlim()
    text_x_shift = 0.005 * (xmax - xmin)

    quantiles = {
        "q5":  q.get(0.05, np.nan),
        "q25": q.get(0.25, np.nan),
        "q50": q.get(0.50, np.nan),
        "q75": q.get(0.75, np.nan),
        "q95": q.get(0.95, np.nan),
    }

    for label, val in quantiles.items():
        if np.isfinite(val):
            ax2.axvline(val, color="black", linestyle="--", linewidth=1.0)
            ax2.text(
                val - text_x_shift,
                text_y_top,
                f"{label}: {val:,.0f}",
                rotation=90,
                ha="right",
                va="top",
                fontsize=7,
                color="black",
            )

    if critical_threshold is not None and np.isfinite(float(critical_threshold)):
        ct = float(critical_threshold)
        ax2.axvline(ct, color=haline_color, linewidth=1.2)
        ax2.text(
            ct - text_x_shift,
            text_y_top,
            f"Critical threshold: {ct:,.0f}",
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            color=haline_color,
        )

    ax2.set_xlabel("Discharge (m³ s⁻¹)", fontsize=7, fontweight="bold")
    ax2.grid(False)
    _despine_axes(ax2)

    #  panel (c): monthly violin 
    ax3 = axes[2]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    raw = [np.asarray(v[v.index.month == m].values, dtype=float) for m in range(1, 13)]
    monthly = [x[np.isfinite(x)] for x in raw]

    positions = [i + 1 for i, x in enumerate(monthly) if x.size > 0]
    data_for_plot = [x for x in monthly if x.size > 0]
    labels_for_plot = [lab for lab, x in zip(month_labels, monthly) if x.size > 0]

    vp = ax3.violinplot(
        data_for_plot,
        positions=positions,
        widths=0.9,
        showmeans=False,
        showextrema=False,
        showmedians=True,
    )

    for body in vp.get("bodies", []):
        body.set_alpha(0.5)
        body.set_facecolor("0.5")
        body.set_edgecolor("0.2")
        body.set_linewidth(0.6)

    if "cmedians" in vp and vp["cmedians"] is not None:
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidths(1.0)

    ax3.set_xticks(positions)
    ax3.set_xticklabels(labels_for_plot, fontsize=7)
    ax3.set_ylabel("Discharge (m³ s⁻¹)", fontsize=7, fontweight="bold")
    if ylim is not None:
        ax3.set_ylim(*ylim)
    ax3.tick_params(labelsize=7)

    if critical_threshold is not None and np.isfinite(float(critical_threshold)):
        ct = float(critical_threshold)
        ax3.axhline(ct, linestyle="-", linewidth=1.0, color=haline_color)
        xmin3, xmax3 = ax3.get_xlim()
        ax3.text(
            xmax3 - 0.4,
            ct,
            f"Critical threshold: {ct:,.1f} m³ s⁻¹",
            ha="right",
            va="bottom",
            fontsize=7,
            color=haline_color,
        )

    ax3.grid(False)
    _despine_axes(ax3)

    for ax, lab in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.text(
            0.01, 0.95, lab,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=8, fontweight="bold",
            color="black",
            zorder=10,
        )

    return fig, axes


# =============================================================================
# Training metrics + test time series 
# =============================================================================

def plot_training_and_timeseries(
    results,
    horizons,
    *,
    date_index,
    y_true,
    pred_df,
    colors_h,
    linestyles_h,
    parameter_label=None,
    roll_window_days=1,
    fig_w=6.9,
    fig_h=3.85,
    dpi=600,
    site=None,
):
    """
    Plot training curves and aligned observed/predicted test time series.
    - Top row: per-epoch curves for selected metrics (train_loss, r2, willmott)
      for all horizons.
    - Bottom row: observed vs predicted time series on the common test date
      intersection.

    Discharge scaling:
    - If parameter_label indicates a 10^4 scaling, observed/predicted values are
      scaled by 1e-4 before plotting to match the label.

    Smoothing:
    - roll_window_days defines a centered rolling window (in days) for display
      smoothing of both observations and predictions.

    **Inputs** :

        results : 'dict'
            Mapping horizon -> loaded run dict (must include "history" and test series).

        horizons : 'iterable'
            Horizons to plot.

        date_index : 'pandas.DatetimeIndex'
            Common test dates (typically from build_aligned_test_series()).

        y_true : 'array-like'
            Observations aligned to date_index.

        pred_df : 'pandas.DataFrame'
            Long-form predictions with columns ["date","horizon","y_pred"].

        colors_h : 'dict'
            Mapping horizon -> color.

        linestyles_h : 'dict'
            Mapping horizon -> linestyle.

        parameter_label : 'str'
            Y-axis label for the time-series panel.

        roll_window_days : 'int'
            Centered rolling-window (days) used for figure smoothing.

        fig_w, fig_h : 'float'
            Figure size in inches.

        dpi : 'int'
            Figure DPI.

        site : 'str or None'
            Optional label printed on the figure (e.g., site id).

    **Outputs** :

        fig : 'matplotlib.figure.Figure'
            Created figure.
    """
    horizons_sorted = sorted(int(h) for h in horizons if int(h) in results)

    candidate_keys = ["train_loss", "r2", "willmott"]
    history_keys = list(next(iter(results.values()))["history"].keys())
    metric_keys = [k for k in candidate_keys if k in history_keys]

    labels_hist = {
        "train_loss": "Train loss ($MSE$)",
        "r2": r"$R^{2}$",
        "willmott": r"Willmott ($d$)",
    }

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    n_metrics = max(1, len(metric_keys))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=n_metrics,
        height_ratios=[1.4, 1.4],
        hspace=0.34,
        wspace=0.325,
    )

    axs_hist = []
    for j in range(n_metrics):
        ax = fig.add_subplot(grid[0, j]) if j == 0 else fig.add_subplot(grid[0, j], sharey=axs_hist[0])
        axs_hist.append(ax)

    panel_letters_top = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for j, mk in enumerate(metric_keys):
        ax = axs_hist[j]
        for h in horizons_sorted:
            hist_vals = results[h]["history"][mk]
            epochs = np.arange(1, len(hist_vals) + 1)
            ax.plot(epochs, hist_vals, lw=1.0, color=colors_h[h], linestyle=linestyles_h[h])

        ax.set_ylabel(labels_hist.get(mk, mk), fontsize=7, fontweight="bold", labelpad=2)
        ax.tick_params(axis="both", labelsize=7)

        if mk in {"r2", "willmott"}:
            ax.set_ylim(0, 1.02)
            ax.set_yticks([0, 0.5, 1.0])

        if j < len(panel_letters_top):
            ax.text(
                0.84, 0.95, panel_letters_top[j],
                transform=ax.transAxes,
                fontsize=7, fontweight="bold",
                ha="left", va="top",
                bbox=dict(facecolor="white", alpha=0.7, pad=1.5, edgecolor="none"),
                zorder=10,
            )

    fig.text(0.51, 0.48, "Epoch", ha="center", va="center", fontsize=7, fontweight="bold")

    ax_ts = fig.add_subplot(grid[1, :])

    y_obs = pd.Series(y_true, index=date_index)
    y_obs_sm = y_obs.rolling(f"{int(roll_window_days)}D", center=True).mean()
    ax_ts.plot(date_index, y_obs_sm, label="Observed", lw=1.3, color="k")

    for h in horizons_sorted:
        sub = pred_df[pred_df["horizon"] == h].sort_values("date")
        y_pred_s = pd.Series(sub["y_pred"].to_numpy(), index=sub["date"]).rolling(
            f"{int(roll_window_days)}D", center=True
        ).mean()
        ax_ts.plot(sub["date"], y_pred_s, lw=1.0, linestyle=linestyles_h[h], color=colors_h[h], alpha=0.7)

    start_pad = pd.to_datetime(date_index.min()) - pd.Timedelta(days=15)
    end_pad = pd.to_datetime(date_index.max()) + pd.Timedelta(days=15)
    ax_ts.set_xlim(start_pad, end_pad)

    ymin = float(np.nanmin(y_obs_sm))
    ymax = float(np.nanmax(y_obs_sm))
    if np.isfinite(ymin) and np.isfinite(ymax):
        pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        ax_ts.set_ylim(ymin - pad, ymax + pad)
    if site is not None:
        ax_ts.set_ylabel(parameter_label, fontsize=7, fontweight="bold", labelpad=8)

    year_locator = mdates.YearLocator()
    year_fmt = mdates.DateFormatter("%Y")
    ax_ts.xaxis.set_major_locator(year_locator)
    ax_ts.xaxis.set_major_formatter(year_fmt)
    ax_ts.tick_params(axis="both", labelsize=7)
    ax_ts.grid(False)

    # Legends
    ax_leg = axs_hist[-1]

    legend_handles_h = [
        Line2D([0], [0], color=colors_h[h], lw=1.0, linestyle=linestyles_h[h], label=f"H={h}")
        for h in horizons_sorted
    ]
    leg1 = ax_leg.legend(
        handles=legend_handles_h,
        labels=[hh.get_label() for hh in legend_handles_h],
        ncol=2,
        frameon=False,
        fontsize=6.5,
        handlelength=1.6,
        columnspacing=0.8,
        borderpad=0.2,
        labelspacing=0.2,
        handletextpad=0.3,
        loc="lower right",
        bbox_to_anchor=(1.02, -0.03),
    )
    leg1.set_title("Lead Time (H)")
    leg1.get_title().set_fontsize(6.5)
    leg1.get_title().set_weight("bold")

    ax_ts.legend(
        handles=[Line2D([0], [0], color="k", lw=1.3, linestyle="-", label="Observed")],
        ncol=1,
        frameon=False,
        fontsize=6.5,
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.2,
        handletextpad=0.3,
        loc="upper right",
        bbox_to_anchor=(0.95, 1.0),
    )

    ax_ts.text(
        0.955, 0.95, "(d)",
        transform=ax_ts.transAxes,
        fontsize=8, fontweight="bold",
        ha="left", va="top",
        bbox=dict(facecolor="white", alpha=0.7, pad=1.5, edgecolor="none"),
        zorder=10,
    )

    if site is not None:
        fig.text(0.125, 0.89, f"Training data: {site}", fontsize=8)
        fig.text(0.125, 0.45, f"Test data: {site}", fontsize=8)
    else:
        fig.text(0.125, 0.89, "Training data", fontsize=8)
        fig.text(0.125, 0.45, "Test data", fontsize=8)

    return fig, ax
