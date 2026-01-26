# -*- coding: utf-8 -*-
"""
figure_creating.py
Example of creating figures for the CNN-LSTM Bonnet Carre Spillway Example
"""

from pathlib import Path
import matplotlib.pyplot as plt

from GaugePredict.routines import get_project_root, resolve_under_project
from GaugePredict.plotting import (
    load_saved_runs,
    parameter_label_from_target,
    build_aligned_test_series,
    get_horizon_styles,
    build_scores_table,
    plot_training_and_timeseries,
    plot_shap_geoplot_grid,
)


# --------------------------------------------------------
# training/test figure
# --------------------------------------------------------
target_site = "01280"
run_name = "function_test"
target_variable = "discharge"
horizons = [1, 3]
site_label = "Bonnet Carré Spillway (USACE)"
save_fig = True

project_root = get_project_root(__file__, levels_up=1)
examples_dir = resolve_under_project(project_root, Path("examples"))
results_base_dir = examples_dir / "results"

results_root = results_base_dir / f"{target_site}_{run_name}"
fig_path = results_root / "fig_training_test_agu.png"

results = load_saved_runs(results_root, horizons, verbose=True)

date_index, y_true, pred_df = build_aligned_test_series(results, horizons)
colors_h, linestyles_h = get_horizon_styles(horizons)
parameter_label = parameter_label_from_target(target_variable)

fig1, ax1 = plot_training_and_timeseries(
    results=results,
    horizons=horizons,
    date_index=date_index,
    y_true=y_true,
    pred_df=pred_df,
    colors_h=colors_h,
    linestyles_h=linestyles_h,
    parameter_label=parameter_label,
    roll_window_days=1,
    site=site_label,
)

if save_fig:
    results_root.mkdir(parents=True, exist_ok=True)
    fig1.savefig(fig_path, dpi=200, bbox_inches="tight")

plt.show()

scores_df = build_scores_table(results, horizons)
print(f"\nTest set scores ({target_site}, {target_variable}):")
print(scores_df.round(3))


# --------------------------------------------------------
# SHAP figure 
# --------------------------------------------------------
target_site = "01280"
shap_run_name = f"{target_site}_test_final"

project_root = get_project_root(__file__, levels_up=1)
examples_dir = resolve_under_project(project_root, Path("examples"))
results_dir = examples_dir / "results"
full_shap_root = results_dir / shap_run_name

horizons = [1, 3]

# currently hardcoded from downsampled model
n_shap_by_h = {
    1: 5,
    3: 9,
    5: 15,
    10: 50,
    15: 90,
    20: 90,
    30: 120,
}

states_fp = examples_dir / "shapefiles" / "US_STATES" / "tl_2023_us_state.shp"

xlim = (-115.0, -76.0)
ylim = (28.5, 50.0)

fig_path = full_shap_root / "fig_shap_agu_paper.png"
save_fig = False

ncols = 2
nrows = 3
fig_w = 5.25
fig_h = 4.75

plot_shap_geoplot_grid(
    shap_root=full_shap_root,
    horizons=horizons,
    n_shap_by_h=n_shap_by_h,
    states_fp=states_fp,
    xlim=xlim,
    ylim=ylim,
    fig_w=fig_w,
    fig_h=fig_h,
    nrows=nrows,
    ncols=ncols,
    s_all=6.0,
    s_used=18.0,
    wspace=0.03,
    hspace=-0.0125,
    cbar_rect=(0.125, 0.07, 0.775, 0.03),
    save_path=fig_path if save_fig else None,
    show=True,
)
