# -*- coding: utf-8 -*-
"""
Provides neural network models and prediction functions for forecasting downstream
gauge conditions. Includes support for training, inference, and SHAP explainability.

Features:
    - Hybrid neural network models for discharge prediction
    - Model training and validation utilities
    - Batch prediction capabilities
    - SHAP value computation for model interpretability
    - PyTorch-based deep learning infrastructure
"""

import json
import os
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import shap

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    Dataset = None
    TORCH_AVAILABLE = False

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from .downloader import load_target
from .routines import (
    build_target_dates,
    generate_full_index,
    generate_sequences,
    generate_train_test_masks,
    load_data,
    load_target_csv,
    process_data,
)


# =============================================================================
# Utils
# =============================================================================
def count_parameters(model):
    """
    Count the number of trainable parameters in PyTorch model.

    **Inputs** :

        model : 'torch.nn.Module'
            Model instance.

    **Outputs** :

        n_params : 'int'
            Total number of parameters with requires_grad=True.
    """
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def get_hardware_info(device_str):
    """
    Collect a hardware and software snapshot for reporting and reproducibility.

    The returned dictionary is intended for logging. If CUDA is requested and
    available, GPU properties are also included.

    **Inputs** :

        device_str : 'str'
            Requested device identifier (commonly "cpu" or "cuda").

    **Outputs** :

        info : 'dict'
            Dictionary containing OS, Python, Torch, CPU, RAM, and optional GPU info.
    """
    info = {
        "device": str(device_str),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": getattr(torch, "__version__", None),
        "cpu_count": os.cpu_count(),
        "ram_gb": psutil.virtual_memory().total / 1024.0**3,
    }

    if TORCH_AVAILABLE and str(device_str) == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_memory_total_gb"] = props.total_memory / 1024.0**3

    return info


# =============================================================================
# Dataset wrapper
# =============================================================================
class SequenceDataset(object):
    """
    Minimal torch.utils.data.Dataset for sequence-to-one regression.

    Expects:
        - X: [N, T, C] float-like array
        - y: [N, 1] or [N] float-like array

    Stores X and y as float32 tensors and exposes `input_channels` for model
    consumption.
    """

    def __init__(self, X, y):
        """
        **Inputs** :

            X : 'array-like'
                Feature tensor with shape [N, T, C].

            y : 'array-like'
                Target array with shape [N] or [N, 1].

        **Outputs** :

            dataset : 'SequenceDataset'
                Dataset containing float32 tensors X and y.
        """
        self._X = np.asarray(X, dtype=np.float32)
        self._y = np.asarray(y, dtype=np.float32)
        self.input_channels = int(self._X.shape[2])

        if TORCH_AVAILABLE:
            try:
                self.X = torch.tensor(self._X, dtype=torch.float32)
                self.y = torch.tensor(self._y, dtype=torch.float32)
            except Exception:
                self.X = self._X
                self.y = self._y
        else:
            self.X = self._X
            self.y = self._y

    def __len__(self):
        """
        Return dataset length.

        **Outputs** :

            n : 'int'
                Number of samples (N).
        """
        return int(self._X.shape[0])

    def __getitem__(self, idx):
        """
        Fetch one sample.

        **Inputs** :

            idx : 'int'
                Sample index.

        **Outputs** :

            X_i, y_i : 'tuple(torch.Tensor, torch.Tensor)'
                Feature sequence [T, C] and target [1] (or scalar-shaped tensor).
        """
        x = self.X[idx]
        y = self.y[idx]
        return x, y


# =============================================================================
# Data Handling and Model Setup
# =============================================================================
class GaugeDataModel:
    """
    Prepare arrays, splits, scalers, and PyTorch DataLoaders for forecasting.
    Future updates will allow for CNN and LSTM separation and layer number configuration
    in notebooks

        Framework:

        - Builds a full daily time index for the modeling window.
        - Loads predictor channels (multiple gauge sites) and a single target series.
        - Converts predictors + target to aligned arrays, then constructs sliding
            sequences for supervised learning.
        - Splits sequences into train/test using a cutoff date.
        - Standardizes predictors per channel using training statistics.
        - Standardizes target using sklearn.StandardScaler.

        Key shapes:

        - processed_X_data: [C, T]
        - X_seq: [N, L, C]   (L = sequence_length)
        - y_seq: [N, 1] or [N]
    """

    def __init__(
        self,
        data_files,
        target_site,
        start_date,
        end_date,
        tz,
        sequence_length,
        forecast_horizon,
        cutoff_date,
        parameter_code,
        *,
        batch_size=64,
        allowed_site_ids_norm=None,
        target_csv_path=None,
        target_csv_date_col=None,
        target_csv_value_col=None,
        target_units="metric",
        target_parameter_kind=None,
    ):
        """
        Configures data pipeline but does not load data yet.
        """
        self.data_files = data_files
        self.target_site = target_site
        self.start_date = start_date
        self.end_date = end_date
        self.tz = tz

        self.sequence_length = int(sequence_length)
        self.forecast_horizon = int(forecast_horizon)
        self.cutoff_date = cutoff_date
        self.parameter_code = parameter_code
        self.batch_size = int(batch_size)

        self.full_index = generate_full_index(start_date, end_date, localize=True, tz=tz)

        if allowed_site_ids_norm:
            self.allowed_site_ids_norm = {str(s).lstrip("0") for s in allowed_site_ids_norm}
        else:
            self.allowed_site_ids_norm = None

        self.target_csv_path = Path(target_csv_path) if target_csv_path else None
        self.target_csv_date_col = target_csv_date_col
        self.target_csv_value_col = target_csv_value_col

        self.target_units = target_units
        self.target_parameter_kind = target_parameter_kind

        self.site_meta_df = None

    def prepare_data(self):
        """
        Load predictors and target and build sequences.

        Framework:
        - Loads raw predictor arrays and per-channel metadata via routines.load_data()
        - Loads the target series via CSV or NWIS
        - Runs routines.process_data() to align and transform predictors
        - Creates supervised sequences X_seq and y_seq via routines.generate_sequences()
        - Builds a metadata table for predictor channels (site id, lat/lon, channel index)

        **Inputs** :
            None (uses instance configuration)

        **Outputs** :
            None (populates instance attributes: processed_X_data, X_seq, y_seq, y, site_meta_df)
        """
        raw_X_data, site_meta = load_data(
            self.data_files,
            self.full_index,
            allow_site_ids_norm=self.allowed_site_ids_norm,
            tz=self.tz,
        )

        if self.target_csv_path is not None:
            target_series = load_target_csv(
                self.target_csv_path,
                self.full_index,
                date_col=self.target_csv_date_col,
                value_col=self.target_csv_value_col,
                tz=self.tz,
            )
        else:
            target_series = load_target(
                self.target_site,
                self.full_index,
                self.start_date,
                self.end_date,
                self.parameter_code,
                to_units=self.target_units,
                tz=self.tz,
                parameter_kind=self.target_parameter_kind,
            )

        self.processed_X_data = process_data(raw_X_data, target_series)

        y = pd.to_numeric(target_series, errors="coerce").to_numpy(dtype=float)
        self.y = y

        self.X_seq, self.y_seq = generate_sequences(
            self.sequence_length,
            self.forecast_horizon,
            self.processed_X_data,
            self.y,
        )

        # predictor channels only
        n_pred_channels = int(self.processed_X_data.shape[0] - 2)
        site_meta = site_meta[:n_pred_channels]

        dfm = pd.DataFrame(site_meta).copy()
        if "site_no" not in dfm.columns:
            dfm["site_no"] = ""
        if "site_no_norm" not in dfm.columns:
            dfm["site_no_norm"] = dfm["site_no"].astype(str).str.lstrip("0")

        dfm["site_no"] = dfm["site_no"].astype(str).fillna("")
        dfm["site_no_norm"] = dfm["site_no_norm"].astype(str).fillna("").str.lstrip("0")

        dfm["lat"] = pd.to_numeric(dfm.get("lat", np.nan), errors="coerce")
        dfm["lon"] = pd.to_numeric(dfm.get("lon", np.nan), errors="coerce")
        dfm["channel_index"] = np.arange(len(dfm), dtype=int)

        self.site_meta_df = dfm.reset_index(drop=True)

    def split_train_test(self):
        """
        Create boolean masks and split sequences into train and test arrays.

        Uses routines.generate_train_test_masks() to create per-sequence masks
        derived from the full daily index, sequence length, forecast horizon, and
        cutoff date. The masks are then applied to X_seq and y_seq.

        **Inputs** :
            None

        **Outputs** :
            None (populates train_mask, test_mask, X_train, y_train, X_test, y_test)
        """
        self.train_mask, self.test_mask = generate_train_test_masks(
            self.full_index,
            self.sequence_length,
            self.y_seq,
            self.forecast_horizon,
            self.cutoff_date,
        )

        self.X_train = self.X_seq[self.train_mask]
        self.y_train = self.y_seq[self.train_mask]
        self.X_test = self.X_seq[self.test_mask]
        self.y_test = self.y_seq[self.test_mask]

    def normalize(self):
        """
        Normalize predictors and targets using training set statistics.

        Predictors (X) are standardized per channel using mean and standard
        deviation computed over the training samples and time dimension.
        Targets (y) are standardized with sklearn.StandardScaler.
        """
        X_train_flat = self.X_train.reshape(-1, self.X_train.shape[-1])
        self.X_mean = np.nanmean(X_train_flat, axis=0)
        self.X_std = np.nanstd(X_train_flat, axis=0)
        self.X_std[self.X_std == 0] = 1.0

        self.X_train_norm = (self.X_train - self.X_mean) / self.X_std
        self.X_test_norm = (self.X_test - self.X_mean) / self.X_std

        self.scaler_y = StandardScaler()
        self.y_train_scaled = self.scaler_y.fit_transform(self.y_train)
        self.y_test_scaled = self.scaler_y.transform(self.y_test)

    def create_datasets(self):
        """
        Wrap normalized arrays in SequenceDataset objects and attach site ids.

        Site ids are stored in both raw and normalized forms:
        - site_no_raw: original strings (may include leading zeros)
        - site_no_norm: leading zeros stripped
        """
        self.train_dataset = SequenceDataset(self.X_train_norm, self.y_train_scaled)
        self.test_dataset = SequenceDataset(self.X_test_norm, self.y_test_scaled)

        ids_raw = self.site_meta_df["site_no"].astype(str).tolist()
        ids_norm = self.site_meta_df["site_no_norm"].astype(str).tolist()

        for ds in (self.train_dataset, self.test_dataset):
            ds.site_ids = ids_norm
            ds.site_ids_raw = ids_raw

        self.site_ids = ids_norm
        self.site_ids_raw = ids_raw

    def setup(self):
        """
        Run the full data preparation pipeline and create DataLoaders.

        **Inputs** :
            None

        **Outputs** :
            None (populates dataloader attributes)
        """
        self.prepare_data()
        self.split_train_test()
        self.normalize()
        self.create_datasets()

        if TORCH_AVAILABLE and DataLoader is not None:
            use_cuda = torch.cuda.is_available()
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=use_cuda,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=use_cuda,
            )
        else:
            # fall back to None for environments without torch
            self.train_dataloader = None
            self.test_dataloader = None


# =============================================================================
# =============================================================================
# Model
# =============================================================================
if TORCH_AVAILABLE and nn is not None:
    class CNN_LSTM(nn.Module):
        """
        CNN-LSTM sequence encoder for regression on daily predictor sequences.
        """

        def __init__(self, input_channels, seq_len):
            super().__init__()

            self.register_buffer("channel_mask", None)

            self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=15, padding=7)
            self.conv2 = nn.Conv1d(128, 128, kernel_size=7, padding=3)
            self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

            self.pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
            self.act = nn.ReLU()
            self.dropout_cnn = nn.Dropout(p=0.05)

            self.lstm = nn.LSTM(
                input_size=128,
                hidden_size=128,
                num_layers=3,
                dropout=0.2,
                batch_first=True,
                bidirectional=False,
            )

            self.dropout_fc = nn.Dropout(p=0.35)
            self.fc = nn.Linear(128, 1)

            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                    nn.init.zeros_(m.bias)

        @torch.no_grad()
        def set_channel_mask(self, mask_1d=None):
            if mask_1d is None:
                self.channel_mask = None
            else:
                self.channel_mask = mask_1d.view(-1)

        def forward(self, x):
            x = x.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]

            if self.channel_mask is not None:
                x = x * self.channel_mask.view(1, -1, 1)

            x = self.act(self.conv1(x))
            x = self.act(self.conv2(x))
            x = self.act(self.conv3(x))

            x = self.dropout_cnn(self.pool(x))

            x = x.permute(0, 2, 1)  # [B, T, 128]
            lstm_out, _ = self.lstm(x)
            last_timestep = lstm_out[:, -1, :]
            return self.fc(self.dropout_fc(last_timestep))
else:
    class CNN_LSTM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not available in the current environment; CNN_LSTM requires torch.")


# =============================================================================
# Metrics
# =============================================================================
def nse_score(y_true, y_pred):
    """
    Nash-Sutcliffe Efficiency (NSE) for regression performance.
    Returns NaN if the denominator is zero (constant observations).

    **Inputs** :

        y_true : 'array-like'
            Observations.

        y_pred : 'array-like'
            Predictions.

    **Outputs** :

        nse : 'float'
            Nash-Sutcliffe efficiency (NaN if undefined).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom


def willmott_score(y_true, y_pred):
    """
    Willmott's index of agreement for regression performance.
    This implementation uses the squared-error form.
    Returns NaN if the denominator is zero.

    **Inputs** :

        y_true : 'array-like'
            Observations.

        y_pred : 'array-like'
            Predictions.

    **Outputs** :

        d : 'float'
            Willmott index of agreement (NaN if undefined).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom


# =============================================================================
# Trainer and evaluation
# =============================================================================
class Trainer:
    """
    Lightweight training and evaluation loop for a PyTorch regression model.

    Features:
    - Device selection (cpu/cuda)
    - Gradient clipping
    - Optional warmup learning-rate scaling
    - Metric evaluation on inverse-transformed targets

    The trainer assumes the target scaler is used to map model outputs and
    dataset targets back to native units for metric computation.
    """

    def __init__(
        self,
        model,
        datamodule,
        scaler_y,
        criterion,
        optimizer,
        *,
        device=None,
        evaluations=None,
        max_grad_norm=1.0,
    ):
        """
        **Inputs** :

            model : 'torch.nn.Module'
                Model to train.

            datamodule : 'GaugeDataModel'
                Prepared data model containing train/test dataloaders.

            scaler_y : 'sklearn.preprocessing.StandardScaler'
                Scaler fit on training targets; used to invert predictions and targets.

            criterion : 'callable'
                Loss function accepting (yhat, y) tensors and returning a scalar loss.

            optimizer : 'torch.optim.Optimizer'
                Optimizer instance.

            device : 'str or None'
                Device to run on ("cpu" or "cuda"). If None, auto-select.

            evaluations : 'dict or None'
                Mapping of metric name -> function(y_true, y_pred).
                Defaults to r2, nse, and willmott.

            max_grad_norm : 'float'
                Maximum gradient norm for clipping.

        **Outputs** :

            Trainer : 'Trainer'
                Configured trainer instance.
        """
        if evaluations is None:
            evaluations = {"r2": r2_score, "nse": nse_score, "willmott": willmott_score}

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available in the current environment; Trainer requires torch.")

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_dataloader = datamodule.train_dataloader
        self.test_dataloader = datamodule.test_dataloader

        self.scaler_y = scaler_y
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluations = evaluations
        self.max_grad_norm = float(max_grad_norm)
        self.history = {}

    def train_epoch(self):
        """
        Train the model for one epoch over the training DataLoader.

        **Inputs** :
            None

        **Outputs** :

            mean_loss : 'float'
                Mean training loss over the epoch, weighted by batch size.
        """
        self.model.train()
        running = 0.0

        for Xb, yb in self.train_dataloader:
            Xb = Xb.to(self.device)
            yb = yb.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            yhat = self.model(Xb)
            loss = self.criterion(yhat, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            running += float(loss.item()) * int(Xb.size(0))

        return running / float(len(self.train_dataloader.dataset))

    def evaluate(self, dataloader=None):
        """
        Run inference on a dataloader and return inverse-transformed y and yhat.

        **Inputs** :

            dataloader : 'torch.utils.data.DataLoader or None'
                DataLoader to evaluate. Defaults to the test DataLoader.

        **Outputs** :

            y_true : 'numpy.ndarray'
                Target values in original (inverse-transformed) units.

            y_pred : 'numpy.ndarray'
                Predicted values in original (inverse-transformed) units.
        """
        if dataloader is None:
            dataloader = self.test_dataloader

        self.model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for Xb, yb in dataloader:
                Xb = Xb.to(self.device)
                yb = yb.to(self.device)
                yhat = self.model(Xb)
                preds.append(yhat.detach().cpu().numpy())
                targets.append(yb.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)

        y_pred = self.scaler_y.inverse_transform(preds)
        y_true = self.scaler_y.inverse_transform(targets)
        return y_true, y_pred

    def fit(
        self,
        num_epochs,
        *,
        evaluate=True,
        warmup_epochs=15,
        warmup_scale_mode="linear",
        warmup_min_scale=0.1,
    ):
        """
        Train at lower learning rate for a fixed number of epochs with optional metric evaluation.

        Framework:
        - If warmup_epochs > 0, learning rates are scaled for early epochs.
        - warmup_scale_mode="linear" scales from 1/warmup_steps up to 1.
        - Other modes use a constant warmup_min_scale.

        **Inputs** :

            num_epochs : 'int'
                Number of training epochs.

            evaluate : 'bool'
                If True, compute metrics each epoch using the test set.

            warmup_epochs : 'int'
                Number of epochs to apply learning-rate warmup.

            warmup_scale_mode : 'str'
                Warmup scaling mode ("linear" or constant fallback).

            warmup_min_scale : 'float'
                Constant scale factor used when warmup_scale_mode is not "linear".

        **Outputs** :

            history : 'dict'
                Training history with keys: "train_loss" and metric names (if enabled).

        **Raises** :

            ValueError
                If warmup_epochs is negative.
        """
        if int(warmup_epochs) < 0:
            raise ValueError("warmup_epochs must be >= 0")

        num_epochs = int(num_epochs)
        warmup_epochs = int(warmup_epochs)

        history = {"train_loss": []}
        if evaluate:
            for name in self.evaluations.keys():
                history[name] = []

        base_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        warmup_steps = min(num_epochs, warmup_epochs) if warmup_epochs > 0 else 0

        for epoch in range(num_epochs):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                if warmup_scale_mode == "linear":
                    scale = float(epoch + 1) / float(max(1, warmup_steps))
                else:
                    scale = float(warmup_min_scale)
                for pg, base_lr in zip(self.optimizer.param_groups, base_lrs):
                    pg["lr"] = float(base_lr) * scale

            train_loss = self.train_epoch()
            history["train_loss"].append(float(train_loss))

            if evaluate:
                y_true, y_pred = self.evaluate()
                y_true = y_true.ravel()
                y_pred = y_pred.ravel()
                for name, func in self.evaluations.items():
                    history[name].append(float(func(y_true, y_pred)))

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        self.history = history
        return history


def evaluate_on_test(trainer, gdm):
    """
    Evaluate a trained model on the GaugeDataModel test split.

    **Inputs** :

        trainer : 'Trainer'
            Trained Trainer instance.

        gdm : 'GaugeDataModel'
            Data model containing test mask and indexing metadata.

    **Outputs** :

        dates : 'pandas.DatetimeIndex or numpy.ndarray'
            Target dates corresponding to each test prediction.

        y_true : 'numpy.ndarray'
            True values in original units.

        y_pred : 'numpy.ndarray'
            Predicted values in original units.

        metrics : 'dict'
            Dictionary with keys: "r2", "nse", "willmott".
    """
    y_true, y_pred = trainer.evaluate(dataloader=gdm.test_dataloader)
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    all_dates = build_target_dates(
        gdm.full_index,
        gdm.sequence_length,
        gdm.forecast_horizon,
        int(len(gdm.y_seq)),
    )
    dates = all_dates[gdm.test_mask]

    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "nse": float(nse_score(y_true, y_pred)),
        "willmott": float(willmott_score(y_true, y_pred)),
    }
    return dates, y_true, y_pred, metrics


# =============================================================================
# SHAP utilities
# =============================================================================
def shap_sites_importance(
    trainer,
    gdm,
    *,
    background_size=32,
    nsamples=512,
    use_test=True,
    random_state=42,
):
    """
    Compute SHAP attributions and aggregate importance to gauge (site) level.

    Framework:

    - Moves the trained model to CPU and sets eval() for SHAP computation.
    - Selects a random subset of training samples for SHAP background.
    - Selects a random subset of samples for evaluation (test set by default).
    - Uses shap.GradientExplainer to compute per-sample attributions.
    - Aggregates absolute SHAP values across samples and timesteps to obtain a
      per-channel importance score.
    - Maps channel importance to site metadata (site_no, lat, lon), then aggregates
      across channels that belong to the same site.

    Expected SHAP shape after conversion: [S, T, C].
    """
    rng = np.random.RandomState(int(random_state))

    model = trainer.model
    orig_device = next(model.parameters()).device
    was_training = model.training

    model.to("cpu")
    model.eval()
    device = torch.device("cpu")

    try:
        n_tr = len(gdm.train_dataset)
        bg_n = int(min(max(16, int(background_size)), n_tr))
        bg_idx = rng.choice(n_tr, size=bg_n, replace=False)
        bg = gdm.train_dataset.X[bg_idx].to(device)

        dataset_eval = (
            gdm.test_dataset
            if (use_test and hasattr(gdm, "test_dataset") and len(gdm.test_dataset) > 0)
            else gdm.train_dataset
        )

        n_ev = len(dataset_eval)
        ev_n = int(min(max(64, int(nsamples)), n_ev))
        ev_idx = rng.choice(n_ev, size=ev_n, replace=False)
        X_ev = dataset_eval.X[ev_idx].to(device)

        explainer = shap.GradientExplainer(model, bg)
        shap_vals = explainer.shap_values(X_ev)

    finally:
        model.to(orig_device)
        model.train() if was_training else model.eval()

    shap_arr = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
    if torch.is_tensor(shap_arr):
        shap_arr = shap_arr.detach().cpu().numpy()

    if shap_arr.ndim != 3:
        raise ValueError(f"Unexpected SHAP shape {shap_arr.shape}; expected [S, T, C]")

    if shap_arr.shape[1] != X_ev.shape[1] or shap_arr.shape[2] != X_ev.shape[2]:
        raise ValueError("SHAP dimensions do not match input dimensions")

    phi_per_channel = np.mean(np.abs(shap_arr), axis=(0, 1))

    n_site_channels = int(gdm.processed_X_data.shape[0] - 2)
    phi_pred = phi_per_channel[:n_site_channels]

    meta = gdm.site_meta_df.iloc[:n_site_channels].copy().reset_index(drop=True)
    if len(meta) != len(phi_pred):
        raise ValueError(f"Metadata rows ({len(meta)}) do not match channels ({len(phi_pred)})")

    meta["importance"] = phi_pred

    agg = (
        meta.groupby("site_no", as_index=False)
        .agg(
            lat=("lat", "first"),
            lon=("lon", "first"),
            n_channels=("channel_index", "count"),
            importance=("importance", "sum"),
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    vmax = float(agg["importance"].max()) if len(agg) else 0.0
    agg["importance_norm"] = 0.0 if vmax <= 0.0 else agg["importance"] / vmax
    agg["rank"] = np.arange(1, len(agg) + 1, dtype=int)

    return agg[["rank", "site_no", "lat", "lon", "importance", "importance_norm", "n_channels"]]


def export_shap_sites(shap_df, out_dir, *, horizon, min_norm_for_list=0.0):
    """
    Export per-site SHAP importance products to CSV and plain-text site list.

    Files written:
    - shap_sites.csv
    - site_list.txt
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = shap_df.copy()
    df["horizon"] = int(horizon)

    cols = ["site_no", "lat", "lon", "importance", "importance_norm", "n_channels", "horizon"]
    df[cols].to_csv(out_dir / "shap_sites.csv", index=False)

    df_list = df if float(min_norm_for_list) <= 0.0 else df[df["importance_norm"] >= float(min_norm_for_list)]

    with open(out_dir / "site_list.txt", "w", encoding="utf-8") as f:
        for s in df_list.sort_values(["importance_norm", "importance"], ascending=False)["site_no"].astype(str):
            f.write(f"{s}\n")

    return out_dir / "shap_sites.csv", out_dir / "site_list.txt"


# =============================================================================
# Horizon directories and SHAP-based site selection
# =============================================================================
def horizon_dir(run_root, h):
    """
    Construct a standardized subdirectory path for a forecast horizon.

    Example:
        horizon_dir("runs", 3) -> Path("runs") / "H03"
    """
    return Path(run_root) / f"H{int(h):02d}"


def load_previous_shap_df(full_shap_root, h):
    """
    Load a previously exported shap_sites.csv for a given horizon if it exists.

    If the CSV exists and contains "site_no" but not "site_no_norm", a normalized
    id column is added by stripping leading zeros.
    """
    shap_csv = horizon_dir(full_shap_root, h) / "shap_sites.csv"
    if not shap_csv.exists():
        return None
    df = pd.read_csv(shap_csv)
    if "site_no" in df.columns and "site_no_norm" not in df.columns:
        df["site_no_norm"] = df["site_no"].astype(str).str.lstrip("0")
    return df


def get_allowed_sites_for_horizon(
    h,
    *,
    shap_root=None,
    site_selection_mode="all",
    n_shap_by_h=None,
    default_n_shap=20,
):
    """
    Return the list of allowed predictor site IDs for a given forecast horizon.

    If ``site_selection_mode`` is "from_shap", this reads the per-horizon
    ``shap_sites.csv`` file (under ``shap_root/Hxx/``) and returns the top sites
    by SHAP importance. Otherwise, returns ``None`` to indicate that all sites
    are allowed.

    Parameters
    ----------
    h : int or str
        Forecast horizon (days ahead).
    shap_root : pathlib.Path or str, optional
        Root directory containing per-horizon SHAP outputs.
    site_selection_mode : str, optional
        Either "from_shap" to limit to top sites or anything else to allow all.
    n_shap_by_h : dict, optional
        Mapping horizon -> number of sites to keep. Falls back to ``default_n_shap``.
    default_n_shap : int, optional
        Default number of sites to keep when ``n_shap_by_h`` is not provided.

    Returns
    -------
    list[str] or None
        List of site_no_norm values (leading zeros stripped) or ``None``.
    """
    if str(site_selection_mode).lower() != "from_shap":
        return None

    if shap_root is None:
        raise ValueError("shap_root must be provided when site_selection_mode='from_shap'")

    shap_csv = horizon_dir(shap_root, h) / "shap_sites.csv"
    if not shap_csv.exists():
        raise FileNotFoundError(
            f"site_selection_mode='from_shap' but shap_sites.csv not found for H={h}: {shap_csv}"
        )

    df = pd.read_csv(shap_csv)

    if "site_no_norm" not in df.columns and "site_no" in df.columns:
        df["site_no_norm"] = df["site_no"].astype(str).str.lstrip("0")

    imp_col = "importance_norm" if "importance_norm" in df.columns else "importance"
    n_sites = int((n_shap_by_h or {}).get(int(h), int(default_n_shap)))

    if imp_col not in df.columns:
        return df.get("site_no_norm", pd.Series(dtype=str)).astype(str).str.lstrip("0").tolist()

    return (
        df.sort_values(imp_col, ascending=False)
        .head(n_sites)["site_no_norm"]
        .astype(str)
        .str.lstrip("0")
        .tolist()
    )


def prepare_shap_sites_used(shap_df_used, *, allowed_sites=None, n_sites=None):
    """
    Normalize and optionally filter a SHAP site table for reporting and saving.

    Framework:
    - Adds site_no_norm if missing
    - Filters to allowed_sites if provided (by site_no_norm)
    - Adds importance_norm if missing (max-normalized importance)
    - Optionally truncates to top n_sites
    - Returns a compact table of commonly used columns
    """
    if shap_df_used is None or len(shap_df_used) == 0:


        return None

    df = shap_df_used.copy()

    if "site_no_norm" not in df.columns and "site_no" in df.columns:
        df["site_no_norm"] = df["site_no"].astype(str).str.lstrip("0")

    if allowed_sites is not None and "site_no_norm" in df.columns:
        df = df[df["site_no_norm"].isin(set(allowed_sites))].copy()

    if "importance_norm" not in df.columns and "importance" in df.columns:
        vals = df["importance"].to_numpy(dtype=float)
        vmax = float(np.nanmax(vals)) if len(vals) else 0.0
        df["importance_norm"] = 0.0 if vmax <= 0.0 else (vals / vmax)

    imp_col = "importance_norm" if "importance_norm" in df.columns else (
        "importance" if "importance" in df.columns else None
    )

    if n_sites is not None and imp_col is not None and df.shape[0] > int(n_sites):
        df = df.sort_values(imp_col, ascending=False).head(int(n_sites)).copy()

    cols_out = [c for c in ["site_no", "site_no_norm", "importance", "importance_norm", "lat", "lon"] if c in df.columns]
    return df[cols_out].copy() if cols_out else None


# =============================================================================
# Save run artifacts and summaries
# =============================================================================
def save_run_artifacts(
    out_dir,
    *,
    dates_test,
    y_true,
    y_pred,
    metrics,
    history,
    model_state_dict,
    scaler_y,
):
    """
    Save standard run outputs to a directory.

    Files written:
    - predictions.csv : date, y_true, y_pred
    - metrics.json    : scalar evaluation metrics
    - history.json    : training history time series
    - model.pt        : torch state dict
    - scaler_y.pkl    : pickled sklearn scaler for inverse transforms
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates_str = pd.to_datetime(dates_test).astype(str)
    pd.DataFrame(
        {"date": dates_str, "y_true": np.asarray(y_true).ravel(), "y_pred": np.asarray(y_pred).ravel()}
    ).to_csv(out_dir / "predictions.csv", index=False)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        out_hist = {}
        for k, v in history.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                out_hist[k] = [float(x) for x in v]
            else:
                out_hist[k] = float(v)
        json.dump(out_hist, f, indent=2)

    torch.save(model_state_dict, out_dir / "model.pt")
    with open(out_dir / "scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)


def update_compute_summary(
    compute_summary,
    *,
    forecast_horizon=None,
    fh=None,  # backward compatible
    n_params=None,
    train_time=None,
    eval_time=None,
    metrics=None,
    hp=None,
):
    """
    Write or update per-horizon results in a compute summary dictionary.

    Stores a record under:

        compute_summary["runs"][forecast_horizon]
    """
    if forecast_horizon is None:
        forecast_horizon = fh
    if forecast_horizon is None:
        raise TypeError("update_compute_summary requires forecast_horizon (or fh)")

    metrics_out = {k: float(v) for k, v in (metrics or {}).items()}

    hp_out = {}
    for k, v in (hp or {}).items():
        if isinstance(v, (int, float)):
            hp_out[k] = v
        elif isinstance(v, np.integer):
            hp_out[k] = int(v)
        elif isinstance(v, np.floating):
            hp_out[k] = float(v)
        elif isinstance(v, np.datetime64):
            hp_out[k] = str(v)
        else:
            hp_out[k] = str(v)

    compute_summary["runs"][int(forecast_horizon)] = {
        "forecast_horizon": int(forecast_horizon),
        "n_parameters": int(n_params) if n_params is not None else None,
        "train_time_sec": float(train_time) if train_time is not None else None,
        "eval_time_sec": float(eval_time) if eval_time is not None else None,
        "metrics": metrics_out,
        "hyperparameters": hp_out,
    }
    return compute_summary


def save_compute_summary(results_root, compute_summary):
    """
    Write a compute summary dictionary to compute_summary.json.
    """
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    with open(results_root / "compute_summary.json", "w", encoding="utf-8") as f:
        json.dump(compute_summary, f, indent=2)


# =============================================================================
# Run per horizon model
# =============================================================================
def run_horizon(
    forecast_horizon,
    *,
    data_files,
    use_csv_target,
    target_site,
    target_parameter_code,
    start_date,
    end_date,
    tz,
    hp,
    device,
    allowed_sites=None,
    csv_path=None,
    csv_date_col=None,
    csv_value_col=None,
    shap_mode="none",
    site_selection_mode="all",
    full_shap_root=None,
    n_sites=None,
    out_dir=None,
    shap_random_state=42,
    target_units="metric",
    target_parameter_kind=None,
):
    """
    Train and evaluate a model for a single forecast horizon, optionally with SHAP.

    Framework:
    - Builds a GaugeDataModel and prepares DataLoaders
    - Instantiates the CNN_LSTM model and optimizer
    - Trains for hp["epochs"] epochs
    - Evaluates on the test split and returns metrics and predictions
    - Optionally computes SHAP site importance ("run") or loads previous SHAP results
    - Optionally writes run artifacts to out_dir
    """
    target_units_nwis = target_units
    if not use_csv_target:
        tu = str(target_units).lower()
        if tu in {"m", "meter", "meters", "metre", "metres"}:
            target_units_nwis = "metric"
        elif tu in {"ft", "feet", "foot"}:
            target_units_nwis = "customary"

    gdm = GaugeDataModel(
        data_files=data_files,
        target_site=None if use_csv_target else target_site,
        start_date=start_date,
        end_date=end_date,
        tz=tz,
        sequence_length=hp["sequence_length"],
        forecast_horizon=int(forecast_horizon),
        cutoff_date=hp["cutoff_date"],
        parameter_code=None if use_csv_target else target_parameter_code,
        batch_size=hp["batch_size"],
        allowed_site_ids_norm=allowed_sites,
        target_csv_path=str(csv_path) if (use_csv_target and csv_path is not None) else None,
        target_csv_date_col=csv_date_col if use_csv_target else None,
        target_csv_value_col=csv_value_col if use_csv_target else None,
        target_units=target_units_nwis,
        target_parameter_kind=target_parameter_kind,
    )
    gdm.setup()

    model = CNN_LSTM(gdm.train_dataset.input_channels, hp["sequence_length"])
    model.dropout_cnn.p = float(hp.get("dropout_cnn", model.dropout_cnn.p))
    model.lstm.dropout = float(hp.get("dropout_lstm", model.lstm.dropout))
    model.dropout_fc.p = float(hp.get("dropout_fc", model.dropout_fc.p))

    n_params = count_parameters(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(hp["learning_rate"]),
        weight_decay=float(hp["weight_decay"]),
    )
    criterion = hp["loss_function"]

    trainer = Trainer(
        model=model,
        datamodule=gdm,
        scaler_y=gdm.scaler_y,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        max_grad_norm=float(hp["max_grad_norm"]),
    )

    t0 = time.perf_counter()
    trainer.fit(int(hp["epochs"]))
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    dates_test, y_true, y_pred, metrics = evaluate_on_test(trainer, gdm)
    eval_time = time.perf_counter() - t1

    if out_dir is not None:
        save_run_artifacts(
            out_dir,
            dates_test=dates_test,
            y_true=y_true,
            y_pred=y_pred,
            metrics=metrics,
            history=trainer.history,
            model_state_dict=trainer.model.state_dict(),
            scaler_y=gdm.scaler_y,
        )

    shap_df_used = None

    if shap_mode == "run":
        shap_df_run = shap_sites_importance(
            trainer,
            gdm,
            background_size=int(hp["background_size"]),
            nsamples=int(hp["nsamples"]),
            use_test=True,
            random_state=shap_random_state,
        )
        if out_dir is not None:
            export_shap_sites(shap_df_run, Path(out_dir), horizon=int(forecast_horizon))
        shap_df_used = shap_df_run.copy()
    else:
        if full_shap_root is not None:
            shap_df_prev = load_previous_shap_df(full_shap_root, int(forecast_horizon))
            if shap_df_prev is not None:
                shap_df_used = shap_df_prev.copy()

    shap_used_out = prepare_shap_sites_used(
        shap_df_used,
        allowed_sites=(allowed_sites if site_selection_mode == "from_shap" else None),
        n_sites=n_sites,
    )

    if shap_used_out is not None and out_dir is not None:
        shap_used_out.to_csv(Path(out_dir) / "shap_sites_used.csv", index=False)

    return {
        "dates_test": dates_test,
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": metrics,
        "history": trainer.history,
        "n_params": n_params,
        "train_time": train_time,
        "eval_time": eval_time,
        "shap_sites_used": shap_used_out,
    }
