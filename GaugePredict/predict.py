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
import cmocean as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from dataretrieval import nwis
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore")
import time as tm
starttm = tm.time()

#  Configuration 
data_dir = Path("cached_data")  # discharge
data_dir_precipitation = Path("cached_data_precipitation")  # precipitation
json_path = data_dir / "site_dict.json"
json_path_precipitation = data_dir_precipitation / "site_dict_precipitation.json"
target_site = '07374000'  # Baton Rouge, LA (2005 - present)
#target_site = '07289000'  # Vicksburg, MS (2008 - present)
# target_site = '07374525'  # Belle Chasse, LA (2009 - present)
start_date, end_date = "2005-01-01", "2025-01-01"
full_index = pd.date_range(start=start_date, end=end_date)
sequence_length = 90  # Change sequence length for days out?
epochs_intial = 150
n_runs = 1
forcast_horizons = [15] #[5, 10, 15]
data_dir.mkdir(exist_ok=True)

#  Load JSON Files 
with open(json_path) as f:
    raw_data_discharge = json.load(f)

with open(json_path_precipitation) as f:
   raw_data_precipitation = json.load(f)

#  Processing Functions (REMOVE THE UTC LATER AND DOWNLOAD WITH UTC AS TZ)
def process_discharge(ts, index):
    ts = pd.Series(ts).replace([-999999, -99999, -9999], np.nan)
    ts.index = pd.to_datetime(ts.index)
    if ts.index.tz is None:
        ts.index = ts.index.tz_localize("UTC")
    else:
        ts.index = ts.index.tz_convert("UTC")
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    ts = ts.sort_index().reindex(index)
    return ts * 0.0283168466  # cfs to m³/s

def process_precipitation(ts, index):
    ts = pd.Series(ts).replace([-999999, -99999, -9999], np.nan)
    ts.index = pd.to_datetime(ts.index)
    if ts.index.tz is None:
        ts.index = ts.index.tz_localize("UTC")
    else:
        ts.index = ts.index.tz_convert("UTC")
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    ts = ts.sort_index().reindex(index)
    return ts * 2.54  # in to cm


#  Process Discharge and Precipitation 
site_dict_discharge = {}
site_dict_precip = {}

for huc_sites in raw_data_discharge.values():
    for site_no, info in huc_sites.items():
        if 'discharge' in info:
            s = process_discharge(info['discharge'], full_index)
            if s.isna().mean() <= 0.25: #remove later
                site_dict_discharge[site_no] = s.interpolate(limit_direction="both").ffill().bfill()

for huc_sites in raw_data_precipitation.values():
    for site_no, info in huc_sites.items():
        if 'precipitation' in info:
            s = process_precipitation(info['precipitation'], full_index)
            if s.isna().mean() <= 0.25: #remove later
               site_dict_precip[site_no] = s.interpolate(limit_direction="both").ffill().bfill()

def load_target():
    df = nwis.get_dv(sites=target_site, parameterCd="00060", start=start_date, end=end_date)[0]
    ts = df['00060_Mean']
    ts.index = pd.to_datetime(ts.index)

    # Localize to UTC only if the index is naive
    if ts.index.tz is None:
        ts.index = ts.index.tz_localize("UTC")
    else:
        ts.index = ts.index.tz_convert("UTC")

    return ts.reindex(full_index.tz_localize("UTC")).interpolate(limit_direction="both").ffill().bfill()


target_discharge = load_target()

#  Construct Feature Stack 
X_data = []

# Discharge sites
for s in site_dict_discharge.values():
    X_data.append(s.reindex_like(target_discharge))

# Target discharge features
X_data.append(target_discharge)
X_data.append(target_discharge.diff().bfill())

# Precipitation sites
for s in site_dict_precip.values():
    X_data.append(s.reindex_like(target_discharge))

# Final stack
X_raw = np.stack([s.values for s in X_data], axis=0)
y = target_discharge.values.copy()

epochs = epochs_intial  # reassigned so I can rerun code without importing data again

# Dataset Class 
class DischargeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
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


# Training 
results = {h: {'r2': [], 'nse': [], 'willmott': [], 'train_loss': []} for h in forcast_horizons}
final_preds = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for forcast_horizon in forcast_horizons:
    print(f" Forecast Horizon: {forcast_horizon} days")
    final_preds[forcast_horizon] = []

    for run in range(n_runs):
        print(f" Run {run + 1}/{n_runs}")

        X_seq, y_seq = [], []
        for i in range(len(y) - sequence_length - forcast_horizon + 1):
            X_seq.append(X_raw[:, i:i + sequence_length].T)
            y_seq.append(y[i + sequence_length + forcast_horizon - 1])  
        
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32).reshape(-1, 1)

        target_dates = full_index[sequence_length : len(full_index) - forcast_horizon + 1]
        target_dates = np.array(target_dates[:len(y_seq)])
        cutoff_date = np.datetime64('2020-01-01')
        train_mask = target_dates < cutoff_date
        test_mask = target_dates >= cutoff_date
        dates = target_dates[test_mask]

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
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = CNN_LSTM(
                input_channels=X_raw.shape[0],
                seq_len=sequence_length).to(device)
        

        optimizer = torch.optim.Adam(model.parameters(), lr=1.15e-6, weight_decay=0.5e-4)  
        criterion = nn.MSELoss() 

        r2_history, nse_history, willmott_history, loss_history = [], [], [], []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)

            avg_loss = epoch_loss / len(train_loader.dataset)

            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    all_preds.append(outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

            y_pred = scaler_y.inverse_transform(np.concatenate(all_preds))
            y_true = scaler_y.inverse_transform(np.concatenate(all_targets))          
            
            r2 = r2_score(y_true.flatten(), y_pred.flatten())
            nse_val = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
            d_index = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2)

            r2_history.append(r2)
            nse_history.append(nse_val)
            willmott_history.append(d_index)
            loss_history.append(avg_loss)


            print(f"Epoch {epoch+1:02d}, Loss: {avg_loss:.4f}, R²: {r2:.4f}, NSE: {nse_val:.4f}, d: {d_index:.4f}")
            
            # Evaluate on training set
            model.eval()
            train_preds, train_targets = [], []
            with torch.no_grad():
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    train_preds.append(outputs.cpu().numpy())
                    train_targets.append(targets.cpu().numpy())
            
            y_pred_train_scaled = np.concatenate(train_preds)
            y_true_train = y_train 
            dates_train = target_dates[train_mask]

        
        best_willmott = -np.inf
        best_preds = None
        
        if d_index > best_willmott:
            best_willmott = d_index
            best_preds = {
                "y_true": y_true,
                "y_pred": y_pred,
                "dates": target_dates[test_mask]
                }

        results[forcast_horizon]['r2'].append(r2_history)
        results[forcast_horizon]['nse'].append(nse_history)
        results[forcast_horizon]['willmott'].append(willmott_history)
        results[forcast_horizon]['train_loss'].append(loss_history)
        
        final_preds[forcast_horizon].append(best_preds)


torch.save(model, f"full_model_{forcast_horizon}_{target_site}")

    

#figures (fix this code later)
# Model Results
fig, axs = plt.subplots(1, 3, figsize=(7, 2), sharex=True, sharey = True, dpi=600)
metrics = ['train_loss', 'r2', 'willmott']
titles = ['Training Loss (MSE)', 'R² Score', 'Willmott Index']
ylabels = [ 'Loss (MSE)', r'Pearson correlation $(R^2)$', r'Willmott index ($d$)']

colors = [cm.cm.haline(0.05), cm.cm.haline(0.2), cm.cm.haline(0.4), cm.cm.haline(0.55), cm.cm.haline(0.65), cm.cm.haline(0.75)]
linestyles = ['--', '-.', ':', '--', '-.', ':']

for i, metric in enumerate(metrics):
    for j, horizon in enumerate(forcast_horizons):
        runs = np.array(results[horizon][metric])  # shape: (n_runs, epochs)
        mean = runs.mean(axis=0)
        std = runs.std(axis=0)
        epochs = np.arange(mean.shape[0])

        axs[i].plot(
            epochs, mean,
            label=f'{horizon}-day',
            color=colors[j % len(colors)],
            linestyle=linestyles[j % len(linestyles)],
            linewidth=1.0
        )
        axs[i].fill_between(
            epochs, mean - std, mean + std,
            color=colors[j % len(colors)], alpha=0.15
        )

    axs[i].set_ylabel(ylabels[i], fontsize=9)
    axs[i].tick_params(labelsize=8)
    axs[i].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    axs[i].xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

axs[0].set_ylim([-0.2, 1.0])
axs[1].set_ylim([-0.2, 1.0])
axs[2].set_ylim([-0.2, 1.0])
axs[1].set_xlabel('Epoch', fontsize=10)

axs[2].legend(
    title='Forecast',
    frameon=False,
    fontsize=8,
    title_fontsize=9,
    handlelength=1.2,
    handletextpad=0.3,
    labelspacing=0.25,
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0)
)

plt.tight_layout(pad=0.45, rect=[0, 0, 0.85, 1])  
plt.show()




## Time Series
colors = [cm.cm.haline(0.05), cm.cm.haline(0.2), cm.cm.haline(0.4), cm.cm.haline(0.55), cm.cm.haline(0.65), cm.cm.haline(0.75)]
linestyles = ['--', '-.', ':', '--', '-.', ':']
def plot_timeseries(preds_dict, key_true, key_pred, key_dates, label, date_start, date_end):
    plt.figure(figsize=(7, 2), dpi=600)
    for idx, horizon in enumerate(forcast_horizons):
        preds = preds_dict[horizon][0]
        y_true = preds[key_true].ravel()
        y_pred = preds[key_pred].ravel()
        dates = pd.to_datetime(preds[key_dates])

        mask = (dates >= date_start) & (dates <= date_end)
        if idx == 0:
            plt.plot(dates[mask], y_true[mask], label="Measured",
                     color="black", linestyle="-", linewidth=1)
            
        plt.plot(dates[mask], y_pred[mask], label=f"{horizon} Day",
                 color=colors[idx], linestyle=linestyles[idx], alpha=0.7, linewidth=1)


    plt.ylabel("Discharge (m³/s)", fontsize=9)
    plt.legend(title='Forecast',
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        handlelength=1.2,
        handletextpad=0.3,
        labelspacing=0.05,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        ncols = 1)
    
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.ylim([0, 1750000])
    plt.xlim([dates[mask].min() - pd.Timedelta(days=5), dates[mask].max() + pd.Timedelta(days=5)])
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.show()


plot_timeseries(
    preds_dict=final_preds,
    key_true="y_true",
    key_pred="y_pred",
    key_dates="dates",
    label="Test",
    date_start="2020-01-01",
    date_end="2025-01-01"
)



plot_timeseries(
    preds_dict=final_preds,
    key_true="y_true",
    key_pred="y_pred",
    key_dates="dates",
    label="Test",
    date_start="2021-01-01",
    date_end="2022-01-01"
)















colors = [cm.cm.haline(0.05), cm.cm.haline(0.2), cm.cm.haline(0.4), cm.cm.haline(0.55), cm.cm.haline(0.65), cm.cm.haline(0.75)]
def plot_correlations(preds_dict, key_true, key_pred, key_dates, title, date_start, date_end):
    fig, axes = plt.subplots(2, 3, figsize=(7, 5), sharex=True, sharey=True, dpi=400)
    axes = axes.flatten()

    all_true, all_pred = [], []
    for i in forcast_horizons:
        preds = preds_dict[i][0]
        y_true = preds[key_true].ravel()
        y_pred = preds[key_pred].ravel()
        dates = pd.to_datetime(preds[key_dates])
        mask = (dates >= date_start) & (dates < date_end)
        all_true.append(y_true[mask])
        all_pred.append(y_pred[mask])

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())

    for idx, i in enumerate(forcast_horizons):
        preds = preds_dict[i][0]
        y_true = preds[key_true].ravel()
        y_pred = preds[key_pred].ravel()
        dates = pd.to_datetime(preds[key_dates])
        mask = (dates >= date_start) & (dates < date_end)
        y_true_sub = y_true[mask]
        y_pred_sub = y_pred[mask]

        ax = axes[idx]
        ax.scatter(y_true_sub, y_pred_sub, color=colors[idx], alpha=0.4, s=8)

        slope, intercept, r_value, _, _ = linregress(y_true_sub, y_pred_sub)
        rmse = np.sqrt(np.mean((y_true_sub - y_pred_sub) ** 2))
        regression_line = slope * y_true_sub + intercept

        ax.plot(y_true_sub, regression_line, color="black", linestyle=":", linewidth=1)
        ax.set_title(f"{i} Day Horizon", fontsize=9)
        ax.set_xlim(min_val, max_val+500000)
        ax.set_ylim(min_val, max_val+500000)
        ax.tick_params(labelsize=9)

        eqn_text = f"$y={slope:.2f}x + {intercept:.2f}$\n$R={r_value:.2f}$"  #\nRMSE={rmse:.2f}
        ax.text(0.05, 0.95, eqn_text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", horizontalalignment="left")

    fig.text(0.5, 0.02, "Measured discharge (m³/s)", ha="center", fontsize=11)
    fig.text(0.02, 0.5, "Predicted discharge (m³/s)", va="center", rotation="vertical", fontsize=11)
    plt.tight_layout(rect=[0.04, 0.04, 1, 1])
    plt.show()


plot_correlations(
    preds_dict=final_preds,
    key_true="y_true",
    key_pred="y_pred",
    key_dates="dates",
    title="Test",
    date_start="2020-01-01",
    date_end="2025-01-01"
)


elapsed = (tm.time() - starttm) / 60
hrs, mins = divmod(elapsed, 60)
print(f"Time Taken: {int(hrs)} hours, {round(mins)} minutes")