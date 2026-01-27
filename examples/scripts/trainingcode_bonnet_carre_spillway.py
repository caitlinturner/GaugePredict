# -*- coding: utf-8 -*-
"""
trainingcode.py
Example use of training the CNN-LSTM Model for Water Level at the Army Corps of Engineers water level gauge
Bonnet Carre Spillway @ Mississippi River (01280)
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from GaugePredict.predict import (
    get_allowed_sites_for_horizon,
    horizon_dir,
    run_horizon,
    save_compute_summary,
    update_compute_summary,
    get_hardware_info,
    TORCH_AVAILABLE,
)

from GaugePredict.routines import get_project_root, resolve_under_project


# --------------------------------------------------------
# Configuration
# --------------------------------------------------------
run_name = "function_test"

site_selection_mode = "from_shap"  # "all" or "from_shap"
shap_mode = "none"  # "none" or "run"

target_site = "01280"
target_variable = "water_level"
target_parameter_code = "30210"
target_units = "metric"

examples_dir = Path("examples")

predictor_type = "discharge"
predictor_json_root = examples_dir / f"cached_data_{predictor_type}"

results_dir = examples_dir / "results"
full_shap_root = results_dir / f"{target_site}_function_test_full"

use_csv_target = True
csv_path = examples_dir / "data" / "bcs_wl.csv"
csv_date_col = "date"
csv_value_col = "wl"

start_date = "2005-01-01"
end_date = "2024-12-31"
tz = "UTC"

horizons = [1, 3]

default_n_shap = 1950
n_shap_by_h = {
    1: 5,
    3: 9,
    5: 15,
    10: 50,
    15: 90,
    20: 90,
    30: 120,
}

hp_defaults = dict(
    sequence_length=60,
    learning_rate=1.0e-4,
    weight_decay=1.0e-3,
    max_grad_norm=1.0,
    dropout_cnn=0.00,
    dropout_lstm=0.20,
    dropout_fc=0.0,
    epochs=25, # Run longer for horizons > 5 (I suggest 100 epochs)
    batch_size=128,
    background_size=32,
    nsamples=128,
    cutoff_date=np.datetime64("2020-01-01"),
    loss_function=None,
)

hp_by_h = {
    1: dict(sequence_length=3, learning_rate=3.0e-4, 
            dropout_cnn=0.00, dropout_lstm=0.00, dropout_fc=0.00,
            weight_decay=5.0e-4, max_grad_norm=1),
    
    3: dict(sequence_length=4, learning_rate=5.0e-5, 
            dropout_cnn=0.00, dropout_lstm=0.00, dropout_fc=0.00,
            weight_decay=3.0e-4, max_grad_norm=1.25),
    
    5: dict(sequence_length=5, learning_rate=3.5e-6, 
            dropout_cnn=0.00, dropout_lstm=0.225, dropout_fc=0.15,
            weight_decay=5.0e-3, max_grad_norm=2.5),
    
    10: dict(sequence_length=6, learning_rate=4.75e-6, 
             dropout_cnn=0.0, dropout_lstm=0.25, dropout_fc=0.30,
             weight_decay=1.0e-2, max_grad_norm=3.25),

    15: dict(sequence_length=10, learning_rate=1.85e-6, 
             dropout_cnn=0.0, dropout_lstm=0.05, dropout_fc=0.375,
             weight_decay=2.5e-2, max_grad_norm=3.5),
    
    20: dict(sequence_length=12, learning_rate=9.5e-7,
             dropout_cnn=0.00, dropout_lstm=0.05, dropout_fc=0.35,
             weight_decay=1.0e-2, max_grad_norm=2.75),

    30: dict(sequence_length=30, learning_rate=7.7e-7,
             dropout_cnn=0.00, dropout_lstm=0.00, dropout_fc=0.35,
             weight_decay=5.25e-2, max_grad_norm=3.75),
}

# Set loss function now that torch is available
if TORCH_AVAILABLE:
    hp_defaults["loss_function"] = torch.nn.MSELoss()



# --------------------------------------------------------
# Resolve paths
# --------------------------------------------------------
project_root = get_project_root(__file__, levels_up=1)

results_root = resolve_under_project(project_root, results_dir / f"{target_site}_{run_name}")
results_root.mkdir(parents=True, exist_ok=True)

predictor_json_root = resolve_under_project(project_root, predictor_json_root)
full_shap_root = resolve_under_project(project_root, full_shap_root)

if use_csv_target:
    csv_path = resolve_under_project(project_root, csv_path)
else:
    csv_path = None

json_path = predictor_json_root / f"site_dict_{predictor_type}.json"

if not json_path.exists():
    raise FileNotFoundError(f"Predictor JSON not found: {json_path}")

if use_csv_target and (csv_path is None or not csv_path.exists()):
    raise FileNotFoundError(f"Target CSV not found: {csv_path}")

data_files = [{"path": json_path, "data_key": "parameter"}]


# --------------------------------------------------------
# Compute summary
# --------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
hardware_info = get_hardware_info(device)

compute_summary = {
    "run_timestamp": datetime.utcnow().isoformat() + "Z",
    "run_name": run_name,
    "target_site": target_site,
    "target_variable": target_variable,
    "target_parameter_code": target_parameter_code,
    "target_units": target_units,
    "predictor_type": predictor_type,
    "start_date": start_date,
    "end_date": end_date,
    "tz": tz,
    "horizons": horizons,
    "site_selection_mode": site_selection_mode,
    "shap_mode": shap_mode,
    "paths": {
        "examples_dir": str(resolve_under_project(project_root, examples_dir)),
        "project_root": str(project_root),
        "results_root": str(results_root),
        "predictor_json_root": str(predictor_json_root),
        "predictor_json_path": str(json_path),
        "full_shap_root": str(full_shap_root),
        "use_csv_target": bool(use_csv_target),
        "target_csv_path": str(csv_path) if use_csv_target else None,
        "target_csv_date_col": csv_date_col if use_csv_target else None,
        "target_csv_value_col": csv_value_col if use_csv_target else None,
    },
    "hardware": hardware_info,
    "runs": {},
}


# --------------------------------------------------------
# Run horizons
# --------------------------------------------------------
for fh in horizons:
    print(f"Horizon H = {fh:02d}")

    hp = dict(hp_defaults)
    if fh in hp_by_h:
        hp.update(hp_by_h[fh])

    n_sites = int(n_shap_by_h.get(fh, default_n_shap))

    allowed_sites = get_allowed_sites_for_horizon(
        fh,
        site_selection_mode=site_selection_mode,
        full_shap_root=full_shap_root,
        n_shap_by_h=n_shap_by_h,
        default_n_shap=default_n_shap,
    )

    out_dir = horizon_dir(results_root, fh)

    out = run_horizon(
        fh,
        data_files=data_files,
        use_csv_target=use_csv_target,
        target_site=target_site,
        target_parameter_code=target_parameter_code,
        start_date=start_date,
        end_date=end_date,
        tz=tz,
        hp=hp,
        device=device,
        allowed_sites=allowed_sites,
        csv_path=csv_path,
        csv_date_col=csv_date_col,
        csv_value_col=csv_value_col,
        shap_mode=shap_mode,
        site_selection_mode=site_selection_mode,
        full_shap_root=full_shap_root,
        n_sites=n_sites,
        out_dir=out_dir,
    )

    compute_summary = update_compute_summary(
        compute_summary,
        fh=fh,
        n_params=out["n_params"],
        train_time=out["train_time"],
        eval_time=out["eval_time"],
        metrics=out["metrics"],
        hp=hp,
    )


save_compute_summary(results_root, compute_summary)
