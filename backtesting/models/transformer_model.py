"""PyTorch Transformer encoder regime classifier.

Causal-style encoder over a fixed-length sequence of monthly features.
Uses standard nn.TransformerEncoderLayer (no flash-attn dependency, since
the input length is small — 12 months by default — so vanilla attention
is fast on the 5090).
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
from backtesting.models.lstm_model import _build_sequences


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class _TransformerNet(nn.Module):
    def __init__(
        self,
        n_features: int,
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = self.norm(x[:, -1, :])
        return self.head(x).squeeze(-1)


class TransformerModel(BaseModel):
    name = "transformer"
    requires_sequences = True

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__(config or ModelConfig())
        self.net: _TransformerNet | None = None
        self.feature_columns: list[str] = []
        self.feature_mean: pd.Series | None = None
        self.feature_std: pd.Series | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _hyper(self) -> dict:
        cfg = self.config
        return {
            "d_model": int(cfg.get("d_model", 64)),
            "nhead": int(cfg.get("nhead", 4)),
            "num_layers": int(cfg.get("num_layers", 2)),
            "dim_feedforward": int(cfg.get("dim_feedforward", 128)),
            "dropout": float(cfg.get("dropout", 0.20)),
            "lr": float(cfg.get("lr", 5e-4)),
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
        d_model = hp["d_model"]
        if d_model % hp["nhead"] != 0:
            d_model = (d_model // hp["nhead"] + 1) * hp["nhead"]
        self.net = _TransformerNet(
            n_features=n_feat,
            d_model=d_model,
            nhead=hp["nhead"],
            num_layers=hp["num_layers"],
            dim_feedforward=hp["dim_feedforward"],
            dropout=hp["dropout"],
            seq_len=seq_len,
        ).to(self.device)
        optim = torch.optim.AdamW(
            self.net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=hp["epochs"])

        train_ds = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq))
        loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True, drop_last=False)

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
            raise RuntimeError("TransformerModel.predict_proba called before fit().")
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


__all__ = ["TransformerModel"]
