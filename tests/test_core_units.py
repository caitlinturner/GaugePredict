# -*- coding: utf-8 -*-
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from GaugePredict.routines import generate_sequences, process_data
from GaugePredict.predict import get_allowed_sites_for_horizon
from GaugePredict.plotting import load_saved_horizon_run, build_scores_table


def test_generate_sequences_shapes_and_alignment():
    # raw predictors: [C, T]
    C, T = 4, 20
    seq_len = 5
    horizon = 3

    x_raw = np.arange(C * T, dtype=np.float32).reshape(C, T)
    y = np.arange(T, dtype=np.float32)

    X_seq, y_seq = generate_sequences(seq_len, horizon, x_raw, y)

    # number of samples = T - seq_len - horizon + 1
    n = T - seq_len - horizon + 1
    assert X_seq.shape == (n, seq_len, C)
    assert y_seq.shape == (n, 1)

    # spot-check alignment:
    # sample i uses x_raw[:, i:i+seq_len] and y at i+seq_len+horizon-1
    i = 2
    assert np.allclose(X_seq[i, :, :].T, x_raw[:, i : i + seq_len])
    assert np.isclose(y_seq[i, 0], y[i + seq_len + horizon - 1])


def test_process_data_appends_two_target_channels():
    C, T = 3, 15
    raw_X = np.random.RandomState(0).randn(C, T).astype(np.float32)

    # target_series length must match T
    target = pd.Series(np.linspace(10.0, 20.0, T, dtype=float))

    Xp = process_data(raw_X, target, smooth_window_days=3)
    assert Xp.shape == (C + 2, T)

    # last 2 channels are target-derived and should be finite
    assert np.isfinite(Xp[-1]).all()
    assert np.isfinite(Xp[-2]).all()


def test_get_allowed_sites_for_horizon_reads_topn(tmp_path: Path):
    # Create fake SHAP file structure like results/<run>/H01/shap_sites.csv
    shap_root = tmp_path / "shap_run"
    h = 1
    hdir = shap_root / f"H{h:02d}"
    hdir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "site_no": ["0001", "0002", "0003", "0004"],
            "importance_norm": [0.1, 0.9, 0.2, 0.8],
            "lat": [0, 0, 0, 0],
            "lon": [0, 0, 0, 0],
        }
    )
    df.to_csv(hdir / "shap_sites.csv", index=False)

    allowed = get_allowed_sites_for_horizon(
        h,
        site_selection_mode="from_shap",
        shap_root=shap_root,
        n_shap_by_h={1: 2},
        default_n_shap=999,
    )

    # top 2 by importance_norm are site_no 0002 and 0004, returned normalized (lstrip zeros)
    assert allowed == ["2", "4"]


def _write_minimal_saved_run(run_root: Path, h: int):
    hdir = run_root / f"H{h:02d}"
    hdir.mkdir(parents=True, exist_ok=True)

    # predictions.csv expected columns: date, y_true, y_pred
    dates = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    pred = pd.DataFrame(
        {
            "date": dates.astype(str),
            "y_true": [1, 2, 3, 4, 5],
            "y_pred": [1.1, 1.9, 3.2, 3.8, 4.9],
        }
    )
    pred.to_csv(hdir / "predictions.csv", index=False)

    # metrics.json
    with open(hdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"r2": 0.5, "nse": 0.4, "willmott": 0.6}, f)

    # history.json
    with open(hdir / "history.json", "w", encoding="utf-8") as f:
        json.dump({"train_loss": [1.0, 0.5], "r2": [0.1, 0.2], "willmott": [0.2, 0.3]}, f)

    # scaler_y.pkl (any sklearn scaler instance is fine for load)
    sc = StandardScaler()
    sc.fit(np.array([0.0, 1.0, 2.0]).reshape(-1, 1))
    with open(hdir / "scaler_y.pkl", "wb") as f:
        pickle.dump(sc, f)

    # model.pt (can be empty bytes; load_saved_horizon_run only checks existence)
    (hdir / "model.pt").write_bytes(b"FAKE_TORCH_STATE_DICT_BYTES")


def test_plotting_load_saved_run_and_scores_table(tmp_path: Path):
    results_root = tmp_path / "results"
    _write_minimal_saved_run(results_root, h=1)
    _write_minimal_saved_run(results_root, h=3)

    run1 = load_saved_horizon_run(results_root, 1, verbose=False)
    assert run1 is not None
    assert len(run1["dates_test"]) == 5
    assert run1["y_true_test"].shape == (5,)
    assert run1["y_pred_test"].shape == (5,)
    assert "metr" in run1 and "history" in run1

    scores = build_scores_table({1: run1, 3: load_saved_horizon_run(results_root, 3, verbose=False)}, [1, 3])
    assert list(scores.columns) == ["r2", "nse", "willmott"]
    assert set(scores.index.tolist()) == {1, 3}
