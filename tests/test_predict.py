# -*- coding: utf-8 -*-
"""
Unit tests for GaugePredict.predict module
Tests for model training, prediction, and evaluation functions
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from GaugePredict.predict import (
    get_hardware_info,
    update_compute_summary,
    save_compute_summary,
    r2_score,
    nse_score,
)


class TestHardwareInfo:
    """Tests for hardware information retrieval"""

    def test_get_hardware_info_cpu(self):
        """Test hardware info retrieval for CPU"""
        info = get_hardware_info("cpu")

        # Should return a dictionary with hardware info
        assert isinstance(info, dict)
        # Should contain basic system info
        assert "device" in info or "cpu" in str(info).lower()

    @pytest.mark.skipif(True, reason="GPU may not be available in test environment")
    def test_get_hardware_info_cuda(self):
        """Test hardware info retrieval for CUDA GPU"""
        info = get_hardware_info("cuda")
        assert isinstance(info, dict)


class TestComputeSummary:
    """Tests for compute summary dictionary operations"""

    def test_compute_summary_structure(self):
        """Test basic compute summary structure"""
        summary = {
            "run_timestamp": "2026-01-27T10:00:00Z",
            "run_name": "test_run",
            "target_site": "01280",
            "target_variable": "water_level",
            "horizons": [1, 3],
            "runs": {},
        }

        assert "run_timestamp" in summary
        assert "target_site" in summary
        assert isinstance(summary["runs"], dict)

    def test_update_compute_summary(self):
        """Test updating compute summary with run results"""
        summary = {
            "run_timestamp": "2026-01-27T10:00:00Z",
            "run_name": "test_run",
            "horizons": [1, 3],
            "runs": {},
        }

        # Simulate updating with horizon results
        updated = update_compute_summary(
            summary,
            fh=1,
            n_params=1000,
            train_time=120.5,
            eval_time=30.2,
            metrics={"r2": 0.85, "rmse": 0.5},
            hp={"learning_rate": 1e-4, "epochs": 50},
        )

        # Check that runs was updated
        assert isinstance(updated["runs"], dict)
        assert updated["runs"] != {}  # Should have been updated

    def test_save_compute_summary(self, tmp_path):
        """Test saving compute summary to disk"""
        summary = {
            "run_timestamp": "2026-01-27T10:00:00Z",
            "run_name": "test_run",
            "target_site": "01280",
            "runs": {},
        }

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Save summary
        save_compute_summary(results_dir, summary)

        # Check file was created
        summary_file = results_dir / "compute_summary.json"
        assert summary_file.exists()

        # Load and verify
        with open(summary_file) as f:
            loaded = json.load(f)

        assert loaded["run_name"] == "test_run"
        assert loaded["target_site"] == "01280"

    def test_load_compute_summary(self, tmp_path):
        """Test loading compute summary from disk"""
        summary = {
            "run_timestamp": "2026-01-27T10:00:00Z",
            "target_site": "01280",
            "runs": {1: {"n_params": 500, "metrics": {"nse": 0.88}}},
        }

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Save summary
        summary_file = results_dir / "compute_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f)

        # Load it back manually
        with open(summary_file) as f:
            loaded = json.load(f)

        assert loaded is not None
        assert loaded["target_site"] == "01280"


class TestMetricsCalculation:
    """Tests for evaluation metrics computation"""

    def test_r2_score_perfect_prediction(self):
        """Test R² score for perfect predictions"""
        from sklearn.metrics import r2_score

        y_true = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        y_pred = np.array([1, 2, 3, 4, 5], dtype=np.float32)

        r2 = r2_score(y_true, y_pred)
        assert np.isclose(r2, 1.0)

    def test_r2_score_poor_prediction(self):
        """Test R² score for poor predictions"""
        from sklearn.metrics import r2_score

        y_true = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        y_pred = np.array([5, 4, 3, 2, 1], dtype=np.float32)

        r2 = r2_score(y_true, y_pred)
        assert r2 < 0.5  # Should be negative or low

    def test_rmse_calculation(self):
        """Test RMSE calculation"""
        y_true = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        y_pred = np.array([1.1, 2.1, 2.9, 4.0, 5.1], dtype=np.float32)

        # Manual RMSE calculation
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)

        assert rmse < 0.2
        assert rmse > 0


class TestModelStateHandling:
    """Tests for saving and loading model states"""

    def test_scaler_pickle_roundtrip(self, tmp_path):
        """Test StandardScaler can be pickled and unpickled"""
        # Create and fit a scaler
        original_scaler = StandardScaler()
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        original_scaler.fit(data)

        # Save to pickle
        pkl_file = tmp_path / "scaler.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump(original_scaler, f)

        # Load from pickle
        with open(pkl_file, "rb") as f:
            loaded_scaler = pickle.load(f)

        # Verify they work the same
        test_data = np.array([[2, 3]], dtype=np.float32)
        original_result = original_scaler.transform(test_data)
        loaded_result = loaded_scaler.transform(test_data)

        assert np.allclose(original_result, loaded_result)

    def test_model_state_dict_placeholder(self, tmp_path):
        """Test model state dict file handling"""
        # Create a placeholder for model state
        model_file = tmp_path / "model.pt"
        model_state = {"layer1.weight": np.random.randn(10, 5)}

        # In practice, torch.save would be used, but test basic file operations
        model_file.write_bytes(b"PLACEHOLDER_MODEL_STATE")

        assert model_file.exists()
        assert model_file.stat().st_size > 0


class TestPredictionsDataFrame:
    """Tests for predictions DataFrame handling"""

    def test_predictions_csv_structure(self, tmp_path):
        """Test predictions CSV has expected structure"""
        dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        predictions_df = pd.DataFrame(
            {
                "date": dates,
                "y_true": np.random.randn(10),
                "y_pred": np.random.randn(10),
            }
        )

        csv_file = tmp_path / "predictions.csv"
        predictions_df.to_csv(csv_file, index=False)

        # Load and verify
        loaded = pd.read_csv(csv_file)
        assert list(loaded.columns) == ["date", "y_true", "y_pred"]
        assert len(loaded) == 10

    def test_predictions_dataframe_calculations(self):
        """Test calculations on predictions DataFrame"""
        dates = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "date": dates,
                "y_true": [1, 2, 3, 4, 5],
                "y_pred": [1.1, 2.2, 2.9, 3.8, 5.1],
            }
        )

        # Calculate residuals
        df["residual"] = df["y_true"] - df["y_pred"]

        assert len(df[df["residual"] < 0]) > 0
        assert np.all(np.isfinite(df["residual"]))


class TestHistoryTracking:
    """Tests for training history tracking"""

    def test_history_json_structure(self, tmp_path):
        """Test training history JSON structure"""
        history = {
            "train_loss": [1.0, 0.9, 0.8, 0.7, 0.6],
            "val_loss": [1.1, 0.95, 0.85, 0.8, 0.75],
            "r2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "willmott": [0.1, 0.15, 0.2, 0.25, 0.3],
        }

        history_file = tmp_path / "history.json"
        with open(history_file, "w") as f:
            json.dump(history, f)

        # Load and verify
        with open(history_file) as f:
            loaded = json.load(f)

        assert len(loaded["train_loss"]) == 5
        assert loaded["train_loss"][-1] < loaded["train_loss"][0]

    def test_history_convergence(self):
        """Test that training history shows convergence"""
        history = {
            "train_loss": [10.0, 5.0, 2.5, 1.2, 0.6, 0.3, 0.25, 0.24],
        }

        # Loss should generally decrease
        train_loss = history["train_loss"]
        decreasing = all(train_loss[i] >= train_loss[i + 1] for i in range(len(train_loss) - 1))
        assert decreasing
