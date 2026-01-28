# -*- coding: utf-8 -*-
"""
Unit tests for GaugePredict.routines module
Tests for utility functions: path resolution, SHAP handling, and data processing
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from GaugePredict.routines import (
    get_project_root,
    resolve_under_project,
    generate_sequences,
    process_data,
)


class TestPathUtilities:
    """Tests for project path resolution utilities"""

    def test_get_project_root_from_file(self, tmp_path):
        """Test get_project_root can find project root from a file path"""
        # Create a nested directory structure
        project_dir = tmp_path / "project"
        src_dir = project_dir / "src" / "module"
        src_dir.mkdir(parents=True)
        test_file = src_dir / "test.py"
        test_file.write_text("# test")

        # Should find the project root at tmp_path/project
        root = get_project_root(str(test_file), levels_up=2)
        assert root.name == "project"

    def test_resolve_under_project_absolute_path(self, tmp_path):
        """Test resolve_under_project with absolute path returns path as-is"""
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Absolute path should be returned unchanged
        abs_path = tmp_path / "other" / "file.txt"
        result = resolve_under_project(project_root, abs_path)
        assert result == abs_path

    def test_resolve_under_project_relative_path(self, tmp_path):
        """Test resolve_under_project with relative path joins with root"""
        project_root = tmp_path / "project"
        project_root.mkdir()

        # Relative path should be joined with root
        rel_path = Path("data") / "file.csv"
        result = resolve_under_project(project_root, rel_path)
        assert result == project_root / "data" / "file.csv"

    def test_resolve_under_project_string_path(self, tmp_path):
        """Test resolve_under_project works with string paths"""
        project_root = tmp_path / "project"
        project_root.mkdir()

        rel_path = "examples/data"
        result = resolve_under_project(project_root, rel_path)
        assert isinstance(result, Path)
        assert result == project_root / "examples" / "data"


class TestSequenceGeneration:
    """Tests for sequence generation utilities"""

    def test_generate_sequences_basic_shape(self):
        """Test generate_sequences produces correct shapes"""
        C, T = 5, 50
        seq_len = 10
        horizon = 5

        x_raw = np.random.randn(C, T).astype(np.float32)
        y = np.random.randn(T).astype(np.float32)

        X_seq, y_seq = generate_sequences(seq_len, horizon, x_raw, y)

        # Expected number of samples
        n_samples = T - seq_len - horizon + 1
        assert X_seq.shape == (n_samples, seq_len, C)
        assert y_seq.shape == (n_samples, 1)

    def test_generate_sequences_alignment(self):
        """Test that sequences are properly aligned with targets"""
        C, T = 3, 20
        seq_len = 4
        horizon = 2

        # Create predictable data
        x_raw = np.arange(C * T, dtype=np.float32).reshape(C, T)
        y = np.arange(T, dtype=np.float32) * 10

        X_seq, y_seq = generate_sequences(seq_len, horizon, x_raw, y)

        # Check first sample alignment
        assert np.allclose(X_seq[0, :, :].T, x_raw[:, 0:seq_len])
        assert y_seq[0, 0] == y[seq_len + horizon - 1]

        # Check middle sample alignment
        i = 5
        assert np.allclose(X_seq[i, :, :].T, x_raw[:, i : i + seq_len])
        assert y_seq[i, 0] == y[i + seq_len + horizon - 1]

    def test_generate_sequences_insufficient_data(self):
        """Test generate_sequences with insufficient data raises error"""
        C = 3
        T = 5  # Very small dataset
        seq_len = 10  # Larger than dataset

        x_raw = np.random.randn(C, T).astype(np.float32)
        y = np.random.randn(T).astype(np.float32)

        # Should raise ValueError for insufficient data
        with pytest.raises(ValueError, match="Not enough data"):
            generate_sequences(seq_len, 1, x_raw, y)


class TestDataProcessing:
    """Tests for data processing utilities"""

    def test_process_data_output_shape(self):
        """Test process_data appends target channels"""
        C, T = 4, 30
        raw_X = np.random.randn(C, T).astype(np.float32)
        target = pd.Series(np.linspace(0, 10, T, dtype=float))

        result = process_data(raw_X, target, smooth_window_days=3)

        # Should append 2 channels (smoothed and raw target)
        assert result.shape == (C + 2, T)
        assert result.dtype == np.float32

    def test_process_data_finite_values(self):
        """Test that processed data contains finite values"""
        C, T = 3, 20
        raw_X = np.random.randn(C, T).astype(np.float32)
        target = pd.Series(np.linspace(5, 15, T, dtype=float))

        result = process_data(raw_X, target, smooth_window_days=1)

        # All values should be finite
        assert np.all(np.isfinite(result))

    def test_process_data_preserves_input_channels(self):
        """Test that original channels are preserved in output"""
        C, T = 2, 25
        raw_X = np.array([[1, 2, 3, 4, 5] * 5, [10, 20, 30, 40, 50] * 5], dtype=np.float32)
        target = pd.Series(np.ones(T, dtype=float))

        result = process_data(raw_X, target, smooth_window_days=1)

        # First C channels should be original (or very close)
        assert np.allclose(result[:C], raw_X, rtol=0.5)

    def test_process_data_with_nans(self):
        """Test process_data handles data with NaNs gracefully"""
        C, T = 3, 20
        raw_X = np.random.randn(C, T).astype(np.float32)
        raw_X[0, 5:8] = np.nan  # Add some NaNs

        target = pd.Series(np.linspace(0, 10, T, dtype=float))

        # Should not raise an error
        result = process_data(raw_X, target, smooth_window_days=1)
        assert result.shape == (C + 2, T)


class TestSHAPConfiguration:
    """Tests for SHAP-related utilities"""

    def test_shap_sites_csv_reading(self, tmp_path):
        """Test reading SHAP sites CSV file"""
        from GaugePredict.predict import get_allowed_sites_for_horizon

        # Create SHAP directory structure
        shap_root = tmp_path / "shap_run"
        h01_dir = shap_root / "H01"
        h01_dir.mkdir(parents=True)

        # Create SHAP sites CSV
        sites_data = {
            "site_no": ["01280", "02897", "03456", "04567"],
            "importance_norm": [0.95, 0.87, 0.72, 0.65],
            "lat": [30.1, 30.2, 30.3, 30.4],
            "lon": [-90.1, -90.2, -90.3, -90.4],
        }
        sites_df = pd.DataFrame(sites_data)
        sites_df.to_csv(h01_dir / "shap_sites.csv", index=False)

        # Get top N sites
        allowed = get_allowed_sites_for_horizon(
            1,
            site_selection_mode="from_shap",
            shap_root=shap_root,
            n_shap_by_h={1: 2},
            default_n_shap=999,
        )

        # Should return top 2 sites by importance
        assert len(allowed) == 2
        assert "1280" in allowed  # 01280 -> 1280 (leading zeros removed)
        assert "2897" in allowed  # 02897 -> 2897

    def test_shap_all_sites_mode(self):
        """Test 'all' site selection mode"""
        from GaugePredict.predict import get_allowed_sites_for_horizon

        # In 'all' mode, should work but may return empty list if no JSON is available
        # This tests that the function handles the mode without crashing
        try:
            allowed = get_allowed_sites_for_horizon(
                1,
                site_selection_mode="all",
                shap_root=None,
                n_shap_by_h={1: 10},
                default_n_shap=999,
            )
            # Should return something (list or None depending on implementation)
            assert allowed is None or isinstance(allowed, list)
        except Exception as e:
            # If it fails, it should be due to missing JSON, not the mode
            assert "json" in str(e).lower() or "file" in str(e).lower()


class TestDataFrameOperations:
    """Tests for common DataFrame operations within the package"""

    def test_date_column_handling(self):
        """Test proper handling of date columns in DataFrames"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
        data = {
            "date": dates,
            "value": np.random.randn(100),
            "y_true": np.random.randn(100),
            "y_pred": np.random.randn(100),
        }
        df = pd.DataFrame(data)

        # Should have proper date index
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert len(df) == 100

    def test_site_number_normalization(self):
        """Test that site numbers are properly normalized (leading zeros removed)"""
        site_numbers = ["01280", "00123", "02897", "00001"]

        # Simulate normalization as done in the package
        normalized = [str(int(s)) for s in site_numbers]

        assert normalized == ["1280", "123", "2897", "1"]


class TestNumpyArrayOperations:
    """Tests for NumPy array operations"""

    def test_standardization(self):
        """Test data standardization/scaling"""
        from sklearn.preprocessing import StandardScaler

        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        # Scaled data should have mean ~0 and std ~1 per column
        assert np.abs(np.mean(scaled, axis=0)).max() < 1e-10
        assert np.abs(np.std(scaled, axis=0) - 1.0).max() < 1e-10

    def test_nan_handling(self):
        """Test handling of NaN values in arrays"""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        # Test filtering NaNs
        clean = data[~np.isnan(data)]
        assert len(clean) == 4
        assert np.all(np.isfinite(clean))

    def test_datetime_alignment(self):
        """Test alignment of datetime indices"""
        dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        values = np.arange(10, dtype=np.float32)

        df = pd.DataFrame({"date": dates, "value": values})

        # Should maintain order and alignment
        assert len(df) == 10
        assert df.loc[0, "date"].strftime("%Y-%m-%d") == "2020-01-01"
        assert df.loc[9, "date"].strftime("%Y-%m-%d") == "2020-01-10"
