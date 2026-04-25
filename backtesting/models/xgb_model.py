"""XGBoost classifier wrapped to fit the BaseModel interface."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from backtesting.models.base import BaseModel, ModelConfig


class XGBModel(BaseModel):
    name = "xgb"
    requires_sequences = False

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__(config or ModelConfig())
        self.model: XGBClassifier | None = None
        self.feature_columns: list[str] = []

    def _hyperparams(self) -> dict:
        cfg = self.config
        return {
            "max_depth": int(cfg.get("max_depth", 3)),
            "learning_rate": float(cfg.get("learning_rate", 0.05)),
            "n_estimators": int(cfg.get("n_estimators", 200)),
            "min_child_weight": int(cfg.get("min_child_weight", 1)),
            "subsample": float(cfg.get("subsample", 0.8)),
            "colsample_bytree": float(cfg.get("colsample_bytree", 0.8)),
            "reg_lambda": float(cfg.get("reg_lambda", 1.0)),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": cfg.seed,
            "tree_method": cfg.get("tree_method", "hist"),
            "verbosity": 0,
        }

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        self.feature_columns = list(X_train.columns)
        eval_set = []
        if X_val is not None and y_val is not None and len(y_val) > 0:
            eval_set = [(X_val.values, y_val.values)]
        self.model = XGBClassifier(**self._hyperparams())
        self.model.fit(X_train.values, y_train.values, eval_set=eval_set, verbose=False)
        self._fitted = True

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted or self.model is None:
            raise RuntimeError("XGBModel.predict_proba called before fit().")
        proba = self.model.predict_proba(X[self.feature_columns].values)[:, 1]
        return pd.Series(proba, index=X.index, dtype=float)


__all__ = ["XGBModel"]
