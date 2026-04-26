"""PyTorch LSTM/Transformer multi-asset return forecasters.

The regime classifier remains XGBoost. These models forecast next-month
asset returns for the Black-Litterman view layer used by the TAA optimizer.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from backtesting.models.base import ModelConfig, standardise
from backtesting.models.transformer_model import _PositionalEncoding

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "benchmark"):
        torch.backends.cudnn.benchmark = True


def _build_return_sequences(
    X: pd.DataFrame,
    y: pd.DataFrame | None,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, list[pd.Timestamp]]:
    """Slice monthly features into sequence windows ending at each row."""
    arr = X.to_numpy(dtype=np.float32)
    n_rows, n_feat = arr.shape
    if n_rows < seq_len:
        empty_x = np.empty((0, seq_len, n_feat), dtype=np.float32)
        return empty_x, None, None, []

    sequences = np.lib.stride_tricks.sliding_window_view(arr, (seq_len, n_feat))[:, 0, :, :]
    sequences = np.ascontiguousarray(sequences, dtype=np.float32)
    sequence_index = list(X.index[seq_len - 1 :])
    if y is None:
        return sequences, None, None, sequence_index

    y_aligned = y.reindex(sequence_index).to_numpy(dtype=np.float32)
    mask = np.isfinite(y_aligned).astype(np.float32)
    y_filled = np.nan_to_num(y_aligned, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return sequences, y_filled, mask, sequence_index


class _ReturnLSTMNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class _ReturnTransformerNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = _PositionalEncoding(d_model, max_len=max(seq_len, 16))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = self.norm(x[:, -1, :])
        return self.head(x)


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    masked = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return masked.sum() / denom


class TorchReturnForecaster:
    """Sequence model for next-month multi-asset return forecasting."""

    def __init__(self, model_type: str, config: ModelConfig | None = None) -> None:
        if model_type not in {"lstm", "transformer"}:
            raise ValueError("model_type must be 'lstm' or 'transformer'.")
        self.model_type = model_type
        self.config = config or ModelConfig()
        self.net: nn.Module | None = None
        self.feature_columns: list[str] = []
        self.target_columns: list[str] = []
        self.feature_mean: pd.Series | None = None
        self.feature_std: pd.Series | None = None
        self.target_mean: pd.Series | None = None
        self.target_std: pd.Series | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fitted = False

    def _hyper(self) -> dict:
        cfg = self.config
        if self.model_type == "lstm":
            return {
                "hidden": int(cfg.get("hidden", 64)),
                "num_layers": int(cfg.get("num_layers", 1)),
                "dropout": float(cfg.get("dropout", 0.15)),
                "lr": float(cfg.get("lr", 8e-4)),
                "weight_decay": float(cfg.get("weight_decay", 1e-4)),
                "epochs": int(cfg.get("epochs", 80)),
                "batch_size": int(cfg.get("batch_size", 32)),
                "patience": int(cfg.get("patience", 12)),
            }
        return {
            "d_model": int(cfg.get("d_model", 48)),
            "nhead": int(cfg.get("nhead", 4)),
            "num_layers": int(cfg.get("num_layers", 1)),
            "dim_feedforward": int(cfg.get("dim_feedforward", 96)),
            "dropout": float(cfg.get("dropout", 0.15)),
            "lr": float(cfg.get("lr", 5e-4)),
            "weight_decay": float(cfg.get("weight_decay", 1e-4)),
            "epochs": int(cfg.get("epochs", 80)),
            "batch_size": int(cfg.get("batch_size", 32)),
            "patience": int(cfg.get("patience", 12)),
        }

    def _standardise_targets(self, y_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        mean = y_train.mean(skipna=True).fillna(0.0)
        std = y_train.std(skipna=True).replace(0.0, 1.0).fillna(1.0)
        return (y_train - mean) / std, mean, std

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
    ) -> None:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        self.feature_columns = list(X_train.columns)
        self.target_columns = list(y_train.columns)
        X_train_n, others_n, mean, std = standardise(
            X_train,
            X_others={"val": X_val} if X_val is not None else None,
        )
        self.feature_mean, self.feature_std = mean, std

        y_train_n, target_mean, target_std = self._standardise_targets(y_train)
        self.target_mean, self.target_std = target_mean, target_std
        y_val_n = None
        if y_val is not None:
            y_val_n = (y_val.reindex(columns=self.target_columns) - target_mean) / target_std

        seq_len = self.config.sequence_length
        X_seq, y_seq, mask_seq, _ = _build_return_sequences(X_train_n, y_train_n, seq_len)
        if len(X_seq) == 0 or y_seq is None or mask_seq is None:
            raise ValueError(f"Train block ({len(X_train)} rows) shorter than sequence_length={seq_len}")

        hp = self._hyper()
        n_feat = X_seq.shape[-1]
        n_out = len(self.target_columns)
        if self.model_type == "lstm":
            self.net = _ReturnLSTMNet(
                n_features=n_feat,
                n_outputs=n_out,
                hidden=hp["hidden"],
                num_layers=hp["num_layers"],
                dropout=hp["dropout"],
            ).to(self.device)
        else:
            d_model = hp["d_model"]
            if d_model % hp["nhead"] != 0:
                d_model = (d_model // hp["nhead"] + 1) * hp["nhead"]
            self.net = _ReturnTransformerNet(
                n_features=n_feat,
                n_outputs=n_out,
                d_model=d_model,
                nhead=hp["nhead"],
                num_layers=hp["num_layers"],
                dim_feedforward=hp["dim_feedforward"],
                dropout=hp["dropout"],
                seq_len=seq_len,
            ).to(self.device)

        optim = torch.optim.AdamW(self.net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=hp["epochs"])
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq), torch.from_numpy(mask_seq)),
            batch_size=hp["batch_size"],
            shuffle=True,
            drop_last=False,
        )

        val_X_seq = val_y_seq = val_mask_seq = None
        if X_val is not None and y_val_n is not None and len(X_val) > 0:
            val_X_seq, val_y_seq, val_mask_seq, _ = _build_return_sequences(others_n["val"], y_val_n, seq_len)

        best_val = math.inf
        best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
        bad_epochs = 0
        for _epoch in range(hp["epochs"]):
            self.net.train()
            for xb, yb, mb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                mb = mb.to(self.device, non_blocking=True)
                optim.zero_grad(set_to_none=True)
                loss = _masked_smooth_l1(self.net(xb), yb, mb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optim.step()
            sched.step()

            if val_X_seq is not None and val_y_seq is not None and val_mask_seq is not None and len(val_X_seq) > 0:
                self.net.eval()
                with torch.no_grad():
                    vx = torch.from_numpy(val_X_seq).to(self.device)
                    vy = torch.from_numpy(val_y_seq).to(self.device)
                    vm = torch.from_numpy(val_mask_seq).to(self.device)
                    val_loss = float(_masked_smooth_l1(self.net(vx), vy, vm).item())
                if val_loss < best_val - 1e-5:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= hp["patience"]:
                        break
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}

        self.net.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted or self.net is None:
            raise RuntimeError("TorchReturnForecaster.predict called before fit().")
        X_n = (X[self.feature_columns] - self.feature_mean) / self.feature_std
        X_seq, _, _, idx = _build_return_sequences(X_n, None, self.config.sequence_length)
        out = pd.DataFrame(np.nan, index=X.index, columns=self.target_columns, dtype=float)
        if len(X_seq) == 0:
            return out.ffill().fillna(0.0)
        self.net.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X_seq).to(self.device)
            pred_n = self.net(xb).cpu().numpy()
        pred = pred_n * self.target_std.to_numpy(dtype=float) + self.target_mean.to_numpy(dtype=float)
        out.loc[idx, self.target_columns] = pred
        return out.ffill().fillna(0.0)


__all__ = ["TorchReturnForecaster"]
