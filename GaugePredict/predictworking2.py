# -*- coding: utf-8 -*-
"""
Flexible temporal models (CNN, LSTM, or hybrid)
predictworking2.py 
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from dataretrieval import nwis


# --------------------------------------------------------
# 
# --------------------------------------------------------


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

def load_target(target_site, full_index):
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
                ext = Path(file).suffix.lower()
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

def get_train_and_test_sets(data_files, target_site, start_date, end_date, conversion_factor, tz, sequence_length, forcast_horizon, cutoff_date, na_filter=0.25):
    full_index = generate_full_index(start_date, end_date, localize=True, tz=tz)
    raw_data = load_data(data_files, full_index, conversion_factor)
    target_discharge = load_target(target_site, full_index)
    processed_data = process_data(raw_data, target_discharge, na_filter=na_filter)
    y =target_discharge.values.copy()
    X_seq, y_seq = generate_sequences(sequence_length, forcast_horizon, processed_data, y)
    train_mask, test_mask = generate_train_test_masks(full_index, sequence_length, y_seq, forcast_horizon, cutoff_date)
    X_train, y_train = X_seq[train_mask], y_seq[train_mask]
    X_test, y_test = X_seq[test_mask], y_seq[test_mask]
    # Normalize X based on training data
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    # Normalize Y
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    train_dataset = DischargeDataset(X_train, y_train_scaled)
    test_dataset = DischargeDataset(X_test, y_test_scaled)
    return train_dataset, test_dataset


# --------------------------------------------------------
# Model definitions
# --------------------------------------------------------
class FlexibleTemporalModel(nn.Module):
    """
    Build one of: 'cnn', 'lstm', 'cnn_lstm'.
    """
    def __init__(self,
                 model_type,
                 input_channels,
                 seq_len,
                 cnn_layers=0,
                 cnn_channels=128,
                 cnn_kernel=3,
                 cnn_dropout=0.2,
                 lstm_hidden=128,
                 lstm_layers=1,
                 lstm_dropout=0.0,
                 bidirectional=False,
                 fc_dropout=0.4,
                 out_features=1):
        super(FlexibleTemporalModel, self).__init__()

        self.model_type = str(model_type).lower()
        self.seq_len = seq_len
        self.relu = nn.ReLU()

        # CNN stack
        self.use_cnn = self.model_type in ("cnn", "cnn_lstm")
        if self.use_cnn:
            padding = int(cnn_kernel // 2)  # preserve length
            convs = []
            in_ch = input_channels
            for _ in range(int(cnn_layers)):
                convs.append(nn.Conv1d(in_channels=in_ch,
                                       out_channels=cnn_channels,
                                       kernel_size=cnn_kernel,
                                       padding=padding))
                in_ch = cnn_channels
            self.cnn = nn.ModuleList(convs)
            self.dropout_cnn = nn.Dropout(p=float(cnn_dropout))
            self.post_cnn_channels = cnn_channels
        else:
            self.cnn = None
            self.dropout_cnn = nn.Identity()
            self.post_cnn_channels = input_channels

        # LSTM block
        self.use_lstm = self.model_type in ("lstm", "cnn_lstm")
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size=self.post_cnn_channels,
                                hidden_size=int(lstm_hidden),
                                num_layers=int(lstm_layers),
                                dropout=float(lstm_dropout) if int(lstm_layers) > 1 else 0.0,
                                batch_first=True,
                                bidirectional=bool(bidirectional))
            lstm_out = int(lstm_hidden) * (2 if bidirectional else 1)
            head_in = lstm_out
        else:
            self.lstm = None
            self.gap = nn.AdaptiveAvgPool1d(output_size=1)
            head_in = self.post_cnn_channels

        # FC head
        self.dropout_fc = nn.Dropout(p=float(fc_dropout))
        self.fc = nn.Linear(in_features=head_in, out_features=int(out_features))

    def forward(self, x):
        # x: [batch, seq_len, input_channels]
        if self.use_cnn:
            x = x.permute(0, 2, 1)     # [B,C,L]
            for conv in self.cnn:
                x = self.relu(conv(x))
            x = self.dropout_cnn(x)
            if self.use_lstm:
                x = x.permute(0, 2, 1) # [B,L,C]

        if self.use_lstm:
            lstm_out, _ = self.lstm(x)               # [B,L,H]
            feats = lstm_out[:, -1, :]
        else:
            feats = self.gap(x).squeeze(-1)          # [B,C]

        out = self.dropout_fc(feats)
        out = self.fc(out)
        return out


# --------------------------------------------------------
# Builders
# --------------------------------------------------------
def build_model(model_type,
                input_channels,
                seq_len,
                cnn_layers=0,
                cnn_channels=128,
                cnn_kernel=3,
                cnn_dropout=0.2,
                lstm_hidden=128,
                lstm_layers=1,
                lstm_dropout=0.2,
                bidirectional=False,
                fc_dropout=0.4,
                out_features=1):
    return FlexibleTemporalModel(model_type=model_type,
                                 input_channels=input_channels,
                                 seq_len=seq_len,
                                 cnn_layers=cnn_layers,
                                 cnn_channels=cnn_channels,
                                 cnn_kernel=cnn_kernel,
                                 cnn_dropout=cnn_dropout,
                                 lstm_hidden=lstm_hidden,
                                 lstm_layers=lstm_layers,
                                 lstm_dropout=lstm_dropout,
                                 bidirectional=bidirectional,
                                 fc_dropout=fc_dropout,
                                 out_features=out_features)


# --------------------------------------------------------
# Training utilities
# --------------------------------------------------------
def evaluate_to_arrays(model, loader, device):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            preds.append(outputs.detach().cpu().numpy())
            targs.append(targets.detach().cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targs, axis=0)
    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    r2 = float(r2_score(y_true.reshape(-1, 1), y_pred.reshape(-1, 1)))
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = float(1.0 - num / den) if den != 0 else np.nan
    num_d = np.sum((y_true - y_pred) ** 2)
    den_d = np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2)
    d_index = float(1.0 - num_d / den_d) if den_d != 0 else np.nan
    return r2, nse, d_index


def train_model(model,
                train_loader,
                test_loader,
                epochs,
                device,
                optimizer,
                criterion,
                scaler_y=None):
    r2_hist, nse_hist, willmott_hist, loss_hist = [], [], [], []
    best_willmott = -np.inf
    best = {"y_true": None, "y_pred": None, "best_willmott": best_willmott}

    for epoch in range(int(epochs)):
        model.train()
        running, nobs = 0.0, 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bs = inputs.size(0)
            running += loss.item() * bs
            nobs += bs

        avg_loss = running / float(max(nobs, 1))
        loss_hist.append(avg_loss)

        y_true, y_pred = evaluate_to_arrays(model, test_loader, device)
        if scaler_y is not None:
            y_true = scaler_y.inverse_transform(y_true)
            y_pred = scaler_y.inverse_transform(y_pred)
        r2, nse, d_index = compute_metrics(y_true, y_pred)
        r2_hist.append(r2)
        nse_hist.append(nse)
        willmott_hist.append(d_index)

        if d_index > best_willmott:
            best_willmott = d_index
            best = {"y_true": y_true, "y_pred": y_pred, "best_willmott": best_willmott}

        print("Epoch %02d, Loss: %.6f, R2: %.4f, NSE: %.4f, d: %.4f"
              % (epoch + 1, avg_loss, r2, nse, d_index))

    histories = {"r2": r2_hist, "nse": nse_hist, "willmott": willmott_hist, "train_loss": loss_hist}
    return histories, best


# --------------------------------------------------------
# BTR setup (CNN+LSTM configuration)
# --------------------------------------------------------
def cnn_lstm_model(input_channels, sequence_length):
    model = build_model(model_type="cnn_lstm",
                        input_channels=input_channels,
                        seq_len=sequence_length,
                        cnn_layers=3,
                        cnn_channels=128,
                        cnn_kernel=3,
                        cnn_dropout=0.2,
                        lstm_hidden=128,
                        lstm_layers=3,
                        lstm_dropout=0.2,
                        bidirectional=False,
                        fc_dropout=0.4,
                        out_features=1)
    first = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=15, padding=7)
    model.cnn[0] = first
    return model


def model_setup(train_loader,
                       test_loader,
                       input_channels,
                       sequence_length,
                       epochs,
                       scaler_y=None,
                       lr=1.15e-6,
                       weight_decay=0.5e-4,
                       save_path=None,
                       device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = cnn_lstm_model(input_channels=input_channels, sequence_length=sequence_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = nn.MSELoss()
    histories, best = train_model(model=model,
                                  train_loader=train_loader,
                                  test_loader=test_loader,
                                  epochs=epochs,
                                  device=device,
                                  optimizer=optimizer,
                                  criterion=criterion,
                                  scaler_y=scaler_y)
    if save_path:
        torch.save(model, save_path)
    return model, histories, best




# Dataset required by get_train_and_test_sets()
class DischargeDataset(Dataset):
    """
    Simple tensor wrapper for (X, y).
    X: [N, seq_len, channels], y: [N, 1]
    """
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _fit_scaler_from_dataset(train_dataset):
    """
    Recreate a y-scaler from the already-scaled training dataset if needed.
    """
    y_arr = train_dataset.y.numpy()
    scaler = StandardScaler(with_mean=False, with_std=False)  # identity placeholder
    # If further scaling is desired later, adjust here.
    return scaler


def make_datasets_and_loaders(data_files,
                              target_site,
                              start_date_str,
                              end_date_str,
                              tz,
                              sequence_length,
                              forcast_horizon,
                              cutoff_date,
                              na_filter=0.25,
                              batch_train=128,
                              batch_test=32,
                              shuffle_train=True):
    """
    Prepare train/test datasets and DataLoaders by calling the collaborator's API.

    Notes:
    - The collaborator's `load_target` uses module-level `start_date`/`end_date`.
      Set those names in the global namespace so we don't alter their function.
    """
    global start_date, end_date
    start_date = start_date_str
    end_date = end_date_str

    train_dataset, test_dataset = get_train_and_test_sets(
        data_files=data_files,
        target_site=target_site,
        start_date=start_date,
        end_date=end_date,
        conversion_factor=1.0,
        tz=tz,
        sequence_length=sequence_length,
        forcast_horizon=forcast_horizon,
        cutoff_date=cutoff_date,
        na_filter=na_filter
    )

    # Rebuild a y scaler (identity placeholder, datasets are already scaled)
    scaler_y = _fit_scaler_from_dataset(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=int(batch_train), shuffle=bool(shuffle_train))
    test_loader = DataLoader(test_dataset, batch_size=int(batch_test), shuffle=False)

    # infer input_channels from one sample: [seq_len, channels]
    sample_x, _ = train_dataset[0]
    input_channels = int(sample_x.shape[-1])

    return train_loader, test_loader, scaler_y, input_channels


def fit_model_generic(train_loader,
                      test_loader,
                      input_channels,
                      sequence_length,
                      model_type="cnn_lstm",
                      cnn_layers=3,
                      lstm_layers=3,
                      epochs=50,
                      lr=1.15e-6,
                      weight_decay=0.5e-4,
                      scaler_y=None,
                      save_path=None,
                      device=None):
    """
    Build and train a model using the flexible builder.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(model_type=model_type,
                        input_channels=int(input_channels),
                        seq_len=int(sequence_length),
                        cnn_layers=int(cnn_layers),
                        cnn_channels=128,
                        cnn_kernel=3,
                        cnn_dropout=0.2,
                        lstm_hidden=128,
                        lstm_layers=int(lstm_layers),
                        lstm_dropout=0.2,
                        bidirectional=False,
                        fc_dropout=0.4,
                        out_features=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    criterion = nn.MSELoss()

    histories, best = train_model(model=model,
                                  train_loader=train_loader,
                                  test_loader=test_loader,
                                  epochs=int(epochs),
                                  device=device,
                                  optimizer=optimizer,
                                  criterion=criterion,
                                  scaler_y=scaler_y)

    if save_path:
        torch.save(model, save_path)

    return model, histories, best


def fit_model_specific(train_loader,
                       test_loader,
                       input_channels,
                       sequence_length,
                       epochs=50,
                       scaler_y=None,
                       save_path=None,
                       device=None):
    """
    Exact CNN+LSTM configuration (first conv k=15, 3 convs, LSTM 3 layers).
    """
    return model_setup(train_loader=train_loader,
                              test_loader=test_loader,
                              input_channels=int(input_channels),
                              sequence_length=int(sequence_length),
                              epochs=int(epochs),
                              scaler_y=scaler_y,
                              save_path=save_path,
                              device=device)


# --------------------------------------------------------
# Notebook Exampple
# --------------------------------------------------------
def run_end_to_end(data_files,
                   target_site,
                   start_date,
                   end_date,
                   tz,
                   sequence_length,
                   forcast_horizon,
                   cutoff_date,
                   na_filter=0.25,
                   batch_train=128,
                   batch_test=32,
                   model_type="cnn_lstm",
                   cnn_layers=3,
                   lstm_layers=3,
                   epochs=50,
                   lr=1.15e-6,
                   weight_decay=0.5e-4,
                   use_specific=False,
                   save_path=None,
                   device=None):
    """
    End-to-end helper for notebooks:
      1) build datasets/loaders from collaborator's functions;
      2) fit either a generic or the specific CNN+LSTM model;
      3) return loaders, model, histories, best, and inferred input_channels.

    **Example**
        train_loader, test_loader, model, histories, best, input_ch = run_end_to_end(
            data_files=data_files,
            target_site='07374000',
            start_date='2005-01-01',
            end_date='2025-01-01',
            tz='UTC',
            sequence_length=90,
            forcast_horizon=15,
            cutoff_date=np.datetime64('2020-01-01'),
            use_specific=True,
            epochs=100,
            save_path='full_model_15_07374000.pt'
        )
    """
    train_loader, test_loader, scaler_y, input_channels = make_datasets_and_loaders(
        data_files=data_files,
        target_site=target_site,
        start_date_str=start_date,
        end_date_str=end_date,
        tz=tz,
        sequence_length=sequence_length,
        forcast_horizon=forcast_horizon,
        cutoff_date=cutoff_date,
        na_filter=na_filter,
        batch_train=batch_train,
        batch_test=batch_test,
        shuffle_train=True
    )

    if use_specific:
        model, histories, best = fit_model_specific(train_loader=train_loader,
                                                    test_loader=test_loader,
                                                    input_channels=input_channels,
                                                    sequence_length=sequence_length,
                                                    epochs=epochs,
                                                    scaler_y=scaler_y,
                                                    save_path=save_path,
                                                    device=device)
    else:
        model, histories, best = fit_model_generic(train_loader=train_loader,
                                                   test_loader=test_loader,
                                                   input_channels=input_channels,
                                                   sequence_length=sequence_length,
                                                   model_type=model_type,
                                                   cnn_layers=cnn_layers,
                                                   lstm_layers=lstm_layers,
                                                   epochs=epochs,
                                                   lr=lr,
                                                   weight_decay=weight_decay,
                                                   scaler_y=scaler_y,
                                                   save_path=save_path,
                                                   device=device)

    return train_loader, test_loader, model, histories, best, input_channels
