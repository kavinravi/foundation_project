"""Abstract model interface.

A model takes a monthly feature frame ``X`` (rows=months, cols=features) and a
binary label series ``y`` indexed identically. After ``fit``, ``predict_proba``
returns the probability that the regime hypothesised by the calling
``Hypothesis`` is active.

For the sequence models (LSTM / Transformer) the model itself maintains the
sliding-window construction so that the rest of the pipeline can stay shape-
agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelConfig:
    """Common knobs the autoresearch agent may tune."""

    seed: int = 42
    sequence_length: int = 12
    extra: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.extra.get(key, default)


class BaseModel(ABC):
    name: str = "base"
    requires_sequences: bool = False

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self._fitted = False

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Return probability of regime=1 indexed identically to ``X``."""

    def predict_signal(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
        signal_when_active: int = -1,
    ) -> pd.Series:
        """Convert probabilities into discrete trade signals.

        ``signal_when_active`` == -1 means "regime=1 → risk-off" (the default
        for stress / two-stage / stagflation); +1 means "regime=1 → risk-on"
        (used by the growth hypothesis).
        """
        probas = self.predict_proba(X)
        signal = np.where(probas >= threshold, signal_when_active, 0)
        return pd.Series(signal, index=probas.index, dtype=int)


def standardise(
    X_train: pd.DataFrame,
    X_others: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.Series, pd.Series]:
    """Z-score normalise on the training mean / std (returned for reuse)."""
    mean = X_train.mean()
    std = X_train.std().replace(0.0, 1.0)
    X_train_n = (X_train - mean) / std
    others_n = {}
    for name, frame in (X_others or {}).items():
        others_n[name] = (frame - mean) / std
    return X_train_n, others_n, mean, std


__all__ = ["BaseModel", "ModelConfig", "standardise"]
