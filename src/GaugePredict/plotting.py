# -*- coding: utf-8 -*-
"""
plotting.py
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
# Helpers
# =============================================================================

def get_examples_results_dir(project_root):
    return Path(project_root) / "examples" / "results"


def parameter_label_from_target(target_variable):
    if str(target_variable).lower() == "discharge":
        return r"Discharge ($10^{4}$ $m^{3}$ $s^{-1}$)"
    return r"Water level (m)"


def horizon_dir(results_root, h):
    return Path(results_root) / f"H{int(h):02d}"


# =============================================================================
# Load saved model outputs
# =============================================================================

def load_saved_horizon_run(results_root, h, *, verbose=True):
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
# SHAP table loading
# =============================================================================

def load_shap_tables_by_horizon(shap_root, horizons, *, filename="shap_sites.csv", verbose=True):
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
    states = gpd.read_file(states_fp)

    if states.crs is None:
        states = states.set_crs(4269)

    states = states.to_crs(4326)

    if "STUSPS" in states.columns:
        states = states[~states["STUSPS"].isin(["AK", "HI", "PR", "VI"])].copy()

    return states


def _normalize_importance_to_unit(df, *, imp_col="importance", out_col="importance_norm"):
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
    csv_path = Path(shap_root) / f"H{int(h):02d}" / "shap_sites.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"SHAP file not found for H={int(h)}: {csv_path}")

    df = pd.read_csv(csv_path)

    for c in ["lat", "lon"]:
        if c not in df.columns:
            raise ValueError(f"{c} column missing in {csv_path}")

    return _normalize_importance_to_unit(df, imp_col="importance", out_col="importance_norm")


def top_n_shap_sites(df, n_keep):
    return df.sort_values("importance_norm", ascending=False).head(int(n_keep)).copy()


# =============================================================================
# SHAP geoplot grid (AGU paper/panel)
# =============================================================================

def plot_shap_geoplot_grid(
    *,
    shap_root,
    horizons,
    n_shap_by_h,
    states_fp,
    xlim,
    ylim,
    fig_w,
    fig_h,
    nrows,
    ncols,
    s_all=6.0,
    s_used=18.0,
    wspace=0.03,
    hspace=-0.0125,
    cbar_rect=(0.125, 0.07, 0.775, 0.03),
    save_path=None,
    show=True,
    dpi=300,
    save_dpi=600,
    font_size=8,
):
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

    return fig


# =============================================================================
# Defaults
# =============================================================================

default_figsize = (7.33, 3.54)
default_dpi = 400

conus_extent = (-14_000_000, -7_300_000, 2_600_000, 7_000_000)
conus_ak_extent = (-20_000_000, -7_000_000, 2_600_000, 11_700_000)


# =============================================================================
# HUC plotting
# =============================================================================

def plot_hucs(
    base_dir,
    states_fp,
    *,
    include_ak=False,
    label_hucs=True,
    basemap=True,
    zoom=4,
):
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

    fig, ax = plt.subplots(figsize=default_figsize, dpi=default_dpi, constrained_layout=True)

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
# Utilities for aligned time series + scores
# =============================================================================

def build_aligned_test_series(results, horizons):
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
    rows = []
    for h in sorted(int(h) for h in horizons if int(h) in results):
        m = results[h]["metr"]
        rows.append({"horizon": int(h), "r2": float(m["r2"]), "nse": float(m["nse"]), "willmott": float(m["willmott"])})
    if not rows:
        return pd.DataFrame(columns=["r2", "nse", "willmott"])
    return pd.DataFrame(rows).set_index("horizon").sort_index()


# =============================================================================
# Training metrics + test time series (AGU-style)
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
    parameter_label=r"Discharge ($10^{4}$ $m^{3}$ $s^{-1}$)",
    roll_window_days=1,
    fig_w=6.9,
    fig_h=3.85,
    dpi=600,
    site=None,
):
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

    return fig
