# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:37:45 2025

@author: cturn
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from dataretrieval import nwis
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore")
import time as tm

def remove_nans(ts):
    return ts.replace([-999999, -99999, -9999], np.nan)

def delocalize_timeseries(ts):
    ts.index = pd.to_datetime(ts.index)
    if ts.index.tz is None:
        ts.index = ts.index.tz_localize("UTC")
    else:
        ts.index = ts.index.tz_convert("UTC")
    return ts

def generate_full_index(start, end, localize=True, tz="UTC"):
    index = pd.date_range(start=start, end=end)
    if localize:
        if index.tz is None:
            index = index.tz_localize(tz)
        else:
            index = index.tz_convert(tz)
    return index

def filter_and_fill_ts(ts, na_filter=0.25):
    """
    Filter out time series with more than na_filter proportion of NaNs.
    Fill remaining NaNs using interpolation and forward/backward fill.
    """
    if ts.isna().mean() > na_filter:
        return None
    ts = ts.interpolate(limit_direction="both").ffill().bfill()
    return ts

def load_target(target_site, full_index, start_date, end_date):
    df = nwis.get_dv(sites=target_site, parameterCd="00060", start=start_date, end=end_date)[0]
    ts = df['00060_Mean']
    ts.index = pd.to_datetime(ts.index)

    # Localize to UTC only if the index is naive
    if ts.index.tz is None:
        ts.index = ts.index.tz_localize("UTC")
    else:
        ts.index = ts.index.tz_convert("UTC")

    return ts.reindex(full_index).interpolate(limit_direction="both").ffill().bfill()

def load_data(data_files, full_index, conversion_factor=1.0):
        data = []
        for file in data_files:
            if type(file) is str:
                Ext = Path(file).suffix.lower()
                if ext in ['.csv', '.txt']:
                    ts = pd.read_csv(file, index_col=0, parse_dates=True)
                elif ext in ['.json']:
                    with open(file, 'r') as f:
                        ts = pd.Series(json.load(f))
            elif type(file) is dict:
                file_path = file['path']
                conversion_factor = file['conversion_factor']
                data_key = file['data_key']
                with open(file_path, 'r') as f:
                    for huc_sites in json.load(f).values():
                        for site_no, info in huc_sites.items():
                            ts = pd.Series(info[data_key])
                            ts = remove_nans(ts)
                            ts = delocalize_timeseries(ts)
                            ts = ts.sort_index().reindex(full_index)
                            ts = ts* conversion_factor  # Apply conversion factor
                            data.append(ts)
        return data
        #return np.array(data)

def process_data(raw_timeseries, target_data, na_filter=0.25):
    processed_data = []
    for data in raw_timeseries:
        if data.isna().mean() <= na_filter:
            processed_data.append(data.interpolate(limit_direction="both").ffill().bfill().reindex_like(target_data))
            processed_data.append(target_data)
            processed_data.append(target_data.diff().bfill())  # Add difference of target discharge
    return np.stack([s.values for s in processed_data], axis=0)

def generate_sequences(sequence_length, forcast_horizon, X_raw, y):
    X_seq, y_seq = [], []
    for i in range(len(y) - sequence_length - forcast_horizon + 1):
        X_seq.append(X_raw[:, i:i + sequence_length].T)
        y_seq.append(y[i + sequence_length + forcast_horizon - 1])  # Adjusted for forecast horizon
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32).reshape(-1, 1)
    return X_seq, y_seq

def generate_train_test_masks(full_index, sequence_length, y_seq, forecast_horizon, cutoff_date):
    target_dates = full_index[sequence_length : len(full_index) - forecast_horizon + 1]
    target_dates = np.array(target_dates[:len(y_seq)])
    cutoff_date = pd.Timestamp(np.datetime64(cutoff_date), tz="UTC")
    train_mask = target_dates < cutoff_date
    test_mask = target_dates >= cutoff_date
    return train_mask, test_mask

class GaugeDataModel:
    """Class to handle data for the CNN-LSTM model."""
    def __init__(self, data_files, target_site, start_date, end_date, tz, sequence_length, forcast_horizon, cutoff_date, batch_size=64, na_filter=0.25):
        self.data_files = data_files
        self.target_site = target_site
        self.start_date = start_date
        self.end_date = end_date
        self.tz = tz
        self.sequence_length = sequence_length
        self.forcast_horizon = forcast_horizon
        self.na_filter = na_filter
        self.full_index = generate_full_index(start_date, end_date, localize=True, tz=tz)
        self.cutoff_date = cutoff_date
        self.batch_size = batch_size

    def prepare_data(self):
        raw_X_data = load_data(self.data_files, self.full_index)
        target_discharge = load_target(self.target_site, self.full_index, self.start_date, self.end_date)
        self.processed_X_data = process_data(raw_X_data, target_discharge, na_filter=self.na_filter)
        self.y = target_discharge.values.copy()
        self.X_seq, self.y_seq = generate_sequences(self.sequence_length, self.forcast_horizon, self.processed_X_data, self.y)

    def split_test_and_train(self):
        self.train_mask, self.test_mask = generate_train_test_masks(self.full_index, self.sequence_length, self.y_seq, self.forcast_horizon, self.cutoff_date)
        self.X_train, self.y_train = self.X_seq[self.train_mask], self.y_seq[self.train_mask]
        self.X_test, self.y_test = self.X_seq[self.test_mask], self.y_seq[self.test_mask]
        # Normalize X based on training data

    def normalize_data(self):
        X_mean = self.X_train.mean(axis=(0, 1), keepdims=True)
        X_std = self.X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        self.X_train_normalized = (self.X_train - X_mean) / X_std
        self.X_test_normalized = (self.X_test - X_mean) / X_std
        # Normalize Y
        self.scaler_y = StandardScaler()
        self.y_train_scaled = self.scaler_y.fit_transform(self.y_train)
        self.y_test_scaled = self.scaler_y.transform(self.y_test)

    def create_datasets(self):
        self.train_dataset = DischargeDataset(self.X_train_normalized, self.y_train_scaled)
        self.test_dataset = DischargeDataset(self.X_test_normalized, self.y_test_scaled)

    def setup(self):
        self.prepare_data()
        self.split_test_and_train()
        self.normalize_data()
        self.create_datasets()
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

# Dataset Class
class DischargeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.input_channels = X.shape[2]
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]



# Model Definitions
class CNN_LSTM(nn.Module):
    def __init__(self, input_channels, seq_len):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=3)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=3)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(p=0.2)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=3,
            dropout=0.2, batch_first=True, bidirectional=False)

        # Fully connected output
        self.dropout_fc = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        # x: [batch_size, seq_len, input_channels]
        x = x.permute(0, 2, 1)  # [batch_size, input_channels, seq_len]

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = self.dropout_cnn(x)

        x = x.permute(0, 2, 1)  # [batch_size, seq_len, 32]
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, 128]
        last_timestep = lstm_out[:, -1, :]  # [batch_size, 128]
        
        out = self.dropout_fc(last_timestep)  # [batch_size, 128]
        out = self.fc(out)  #  [batch_size, 1]

        return out

def nse_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def willmott_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2)

class Trainer:

    def __init__(self, model, datamodule, scaler_y, criterion, optimizer, device=None, evaluations = {'r2': r2_score,
                                                                                                      'nse': nse_score,
                                                                                                      'willmott': willmott_score}):
        self.model = model
        self.train_dataloader = datamodule.train_dataloader
        self.test_dataloader = datamodule.test_dataloader
        self.scaler_y = scaler_y
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluations = evaluations

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for X_batch, y_batch in self.train_dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        avg_loss = running_loss / len(self.train_dataloader.dataset)
        return avg_loss

    def evaluate(self, dataloader=None):
        if dataloader is None:
            dataloader = self.test_dataloader
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        all_preds_rescaled = self.scaler_y.inverse_transform(np.concatenate(all_preds))
        all_targets_rescaled = self.scaler_y.inverse_transform(np.concatenate(all_targets))
        return all_targets_rescaled, all_preds_rescaled

    def fit(self, num_epochs, evalulate=True):
        history = {'train_loss': []}
        if evalulate:
            history.update({eval:[] for eval in self.evaluations.keys()})
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = self.train_epoch()
            history['train_loss'].append(train_loss)
            if evalulate:
                targets, preds = self.evaluate()
                for eval_name, eval_func in self.evaluations.items():
                    score = eval_func(targets, preds)
                    history[eval_name].append(score)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        self.history = history
        return history
