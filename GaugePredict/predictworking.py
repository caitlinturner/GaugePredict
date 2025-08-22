# -*- coding: utf-8 -*-
"""
Flexible temporal models (CNN, LSTM, or hybrid) and a callable training setup.
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score


# --------------------------------------------------------
# Model definitions
# --------------------------------------------------------
class FlexibleTemporalModel(nn.Module):
    """
    Build one of: 'cnn', 'lstm', 'cnn_lstm'.

    **Inputs** :
        model_type : 'cnn' | 'lstm' | 'cnn_lstm'
        input_channels : int
        seq_len : int
        cnn_layers : int            (ignored if model_type == 'lstm')
        cnn_channels : int          (per conv layer)
        cnn_kernel : int            (same kernel for all convs, padding='same')
        cnn_dropout : float
        lstm_hidden : int           (ignored if model_type == 'cnn')
        lstm_layers : int
        lstm_dropout : float
        bidirectional : bool
        fc_dropout : float
        out_features : int

    **Outputs** :
        forward(x) -> tensor of shape [batch, out_features]
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
            self.post_cnn_channels = input_channels  # passes straight to LSTM or head

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
            self.seq_reduce = "lstm_last"  # take last timestep from LSTM
        else:
            self.lstm = None
            # for CNN-only, reduce over time by adaptive pooling
            self.gap = nn.AdaptiveAvgPool1d(output_size=1)
            head_in = self.post_cnn_channels
            self.seq_reduce = "gap"

        # FC head
        self.dropout_fc = nn.Dropout(p=float(fc_dropout))
        self.fc = nn.Linear(in_features=head_in, out_features=int(out_features))

    def forward(self, x):
        # x: [batch, seq_len, input_channels]
        if self.use_cnn:
            # to [batch, channels, seq]
            x = x.permute(0, 2, 1)
            for conv in self.cnn:
                x = self.relu(conv(x))
            x = self.dropout_cnn(x)

            if self.use_lstm:
                # back to [batch, seq, channels] for LSTM
                x = x.permute(0, 2, 1)
        else:
            # no CNN, ensure shape for LSTM if needed
            pass

        if self.use_lstm:
            lstm_out, _ = self.lstm(x)               # [batch, seq, hidden]
            feats = lstm_out[:, -1, :]               # last timestep
        else:
            # CNN-only: GAP over time dimension (seq length axis)
            feats = self.gap(x).squeeze(-1)          # [batch, channels]

        out = self.dropout_fc(feats)
        out = self.fc(out)
        return out


# --------------------------------------------------------
# Model builders 
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
    """Return a FlexibleTemporalModel with requested components."""
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
    """Run model on a loader and return concatenated numpy arrays (pred, target)."""
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
    """Return R², NSE, Willmott d (arrays are unscaled)."""
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    r2 = float(r2_score(y_true, y_pred))

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
    """
    Train and evaluate per epoch. If scaler_y is provided, inverse-transform outputs and targets.

    **Outputs** :
        histories : dict with 'r2','nse','willmott','train_loss'
        best : dict with 'y_true','y_pred','best_willmott'
    """
    r2_hist, nse_hist, willmott_hist, loss_hist = [], [], [], []
    best_willmott = -np.inf
    best = {"y_true": None, "y_pred": None, "best_willmott": best_willmott}

    for epoch in range(int(epochs)):
        model.train()
        running = 0.0
        nobs = 0

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

        # evaluation on test set
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

    histories = {
        "r2": r2_hist,
        "nse": nse_hist,
        "willmott": willmott_hist,
        "train_loss": loss_hist,
    }
    return histories, best


# --------------------------------------------------------
# Specific setup (replicates your CNN+LSTM with 3 conv, 3-layer LSTM, 128 width)
# --------------------------------------------------------
def make_cnn_lstm_default(input_channels, sequence_length):
    """
    CNN+LSTM with:
      - 3 Conv1d layers, 128 channels, kernel sizes ~'same' (k=15 for first, then k=3)
      - LSTM hidden=128, num_layers=3, dropout=0.2
      - FC dropout=0.4, output=1
    """
    # Build using uniform kernel, then override first conv to k=15 like your snippet
    model = build_model(model_type="cnn_lstm",
                        input_channels=input_channels,
                        seq_len=sequence_length,
                        cnn_layers=3,
                        cnn_channels=128,
                        cnn_kernel=3,        # will adjust first conv to 15 below
                        cnn_dropout=0.2,
                        lstm_hidden=128,
                        lstm_layers=3,
                        lstm_dropout=0.2,
                        bidirectional=False,
                        fc_dropout=0.4,
                        out_features=1)

    # Adjust first conv to kernel_size=15, padding=7 (preserve length)
    first = nn.Conv1d(in_channels=input_channels,
                      out_channels=128,
                      kernel_size=15,
                      padding=7)
    model.cnn[0] = first
    return model


def run_specific_setup(train_loader,
                       test_loader,
                       input_channels,
                       sequence_length,
                       epochs,
                       scaler_y=None,
                       lr=1.15e-6,
                       weight_decay=0.5e-4,
                       save_path=None,
                       device=None):
    """
    Build the specific CNN+LSTM model, train, evaluate, and optionally save.

    **Inputs** :
        train_loader, test_loader : DataLoader
        input_channels : int  (e.g., X_raw.shape[0])
        sequence_length : int
        epochs : int
        scaler_y : optional sklearn scaler with inverse_transform
        lr, weight_decay : optimizer params
        save_path : str or None  (torch.save path)
        device : torch.device or None

    **Outputs** :
        model, histories, best
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = make_cnn_lstm_default(input_channels=input_channels,
                                  sequence_length=sequence_length).to(device)

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

