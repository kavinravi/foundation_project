"""PyTorch LSTM regime classifier.

Targets the RTX 5090 (Blackwell sm_120). Uses standard nn.LSTM, AdamW,
cosine LR schedule, and mixed-precision training where supported.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from backtesting.models.base import BaseModel, ModelConfig, standardise


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, num_layers: int, dropout: float) -> None:
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
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _build_sequences(
    X: pd.DataFrame, y: pd.Series | None, seq_len: int
) -> tuple[np.ndarray, np.ndarray | None, list[pd.Timestamp]]:
    """Slice ``X`` into shape (N, seq_len, n_features) windows ending at each row."""
    arr = X.to_numpy(dtype=np.float32)
    n_rows, n_feat = arr.shape
    if n_rows < seq_len:
        return np.empty((0, seq_len, n_feat), dtype=np.float32), None, []

    sequences = np.lib.stride_tricks.sliding_window_view(arr, (seq_len, n_feat))[:, 0, :, :]
    sequence_index = list(X.index[seq_len - 1 :])
    if y is not None:
        labels = y.reindex(sequence_index).to_numpy(dtype=np.float32)
        return sequences.astype(np.float32, copy=False), labels, sequence_index
    return sequences.astype(np.float32, copy=False), None, sequence_index


class LSTMModel(BaseModel):
    name = "lstm"
    requires_sequences = True

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__(config or ModelConfig())
        self.net: _LSTMNet | None = None
        self.feature_columns: list[str] = []
        self.feature_mean: pd.Series | None = None
        self.feature_std: pd.Series | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _hyper(self) -> dict:
        cfg = self.config
        return {
            "hidden": int(cfg.get("hidden", 96)),
            "num_layers": int(cfg.get("num_layers", 2)),
            "dropout": float(cfg.get("dropout", 0.20)),
            "lr": float(cfg.get("lr", 1e-3)),
            "weight_decay": float(cfg.get("weight_decay", 1e-4)),
            "epochs": int(cfg.get("epochs", 200)),
            "batch_size": int(cfg.get("batch_size", 32)),
            "patience": int(cfg.get("patience", 30)),
            "pos_weight": float(cfg.get("pos_weight", 0.0)) or None,
        }

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        self.feature_columns = list(X_train.columns)
        X_train_n, others_n, mean, std = standardise(
            X_train,
            X_others={"val": X_val} if X_val is not None else None,
        )
        self.feature_mean, self.feature_std = mean, std

        seq_len = self.config.sequence_length
        X_seq, y_seq, _ = _build_sequences(X_train_n, y_train, seq_len)
        if len(X_seq) == 0:
            raise ValueError(
                f"Train block ({len(X_train)} rows) shorter than sequence_length={seq_len}"
            )

        hp = self._hyper()
        n_feat = X_seq.shape[-1]
        self.net = _LSTMNet(n_feat, hp["hidden"], hp["num_layers"], hp["dropout"]).to(self.device)
        optim = torch.optim.AdamW(
            self.net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=hp["epochs"])

        train_ds = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq))
        loader = DataLoader(
            train_ds,
            batch_size=hp["batch_size"],
            shuffle=True,
            drop_last=False,
        )

        val_X_seq = val_y_seq = None
        if X_val is not None and y_val is not None:
            val_X_seq, val_y_seq, _ = _build_sequences(others_n["val"], y_val, seq_len)

        pos_weight_value = hp["pos_weight"]
        if pos_weight_value is None and y_seq.sum() > 0 and y_seq.sum() < len(y_seq):
            pos_weight_value = float((len(y_seq) - y_seq.sum()) / max(y_seq.sum(), 1.0))
        pos_weight_tensor = (
            torch.tensor(pos_weight_value, dtype=torch.float32, device=self.device)
            if pos_weight_value
            else None
        )

        best_val_loss = math.inf
        best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
        bad_epochs = 0
        for _epoch in range(hp["epochs"]):
            self.net.train()
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optim.zero_grad(set_to_none=True)
                logits = self.net(xb)
                loss = F.binary_cross_entropy_with_logits(
                    logits, yb, pos_weight=pos_weight_tensor
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optim.step()
            sched.step()

            if val_X_seq is not None and len(val_X_seq) > 0:
                self.net.eval()
                with torch.no_grad():
                    vx = torch.from_numpy(val_X_seq).to(self.device)
                    vy = torch.from_numpy(val_y_seq).to(self.device)
                    val_loss = float(F.binary_cross_entropy_with_logits(self.net(vx), vy).item())
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
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

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted or self.net is None:
            raise RuntimeError("LSTMModel.predict_proba called before fit().")
        X_n = (X[self.feature_columns] - self.feature_mean) / self.feature_std
        X_seq, _, idx = _build_sequences(X_n, None, self.config.sequence_length)
        if len(X_seq) == 0:
            return pd.Series(dtype=float, index=X.index)
        self.net.eval()
        with torch.no_grad():
            t = torch.from_numpy(X_seq).to(self.device)
            logits = self.net(t)
            probs = torch.sigmoid(logits).cpu().numpy()
        full = pd.Series(np.nan, index=X.index, dtype=float)
        full.loc[idx] = probs
        return full.ffill().fillna(0.5)


__all__ = ["LSTMModel"]
