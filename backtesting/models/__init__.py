"""Predictive models: XGBoost, PyTorch LSTM, PyTorch Transformer."""

from backtesting.models.base import BaseModel, ModelConfig
from backtesting.models.xgb_model import XGBModel
from backtesting.models.lstm_model import LSTMModel
from backtesting.models.return_forecaster import TorchReturnForecaster
from backtesting.models.transformer_model import TransformerModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "xgb": XGBModel,
    "lstm": LSTMModel,
    "transformer": TransformerModel,
}


def build_model(name: str, **kwargs) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Choose from {sorted(MODEL_REGISTRY)}.")
    return MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "BaseModel",
    "LSTMModel",
    "MODEL_REGISTRY",
    "ModelConfig",
    "TorchReturnForecaster",
    "TransformerModel",
    "XGBModel",
    "build_model",
]
