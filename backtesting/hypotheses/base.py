"""Base ``Hypothesis`` interface.

Each hypothesis declares:

* ``feature_columns`` — names of the columns it will produce.
* ``compute_features`` — turns the joint price / signal panel into a
  monthly-frequency feature frame.
* ``compute_label`` — produces a binary label series indicating whether
  the hypothesised regime is active in the next month.
* ``signal_when_active`` — ``-1`` (risk-off) or ``+1`` (risk-on).
* ``tilt`` — additive per-asset tilt applied when the signal fires.

The framework is intentionally feature-agnostic so that the
autoresearch agent can edit any of these in ``train.py`` without
touching the core engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtesting.core.data import (
    MarketPanel,
    monthly_resample,
    relative_momentum,
    trailing_window_change,
    trend_zscore,
)


@dataclass
class HypothesisMeta:
    name: str
    title: str
    summary: str
    signal_when_active: int
    tilt: dict[str, float] = field(default_factory=dict)


class Hypothesis(ABC):
    meta: HypothesisMeta
    feature_columns: list[str]

    @abstractmethod
    def compute_features(self, panel: MarketPanel) -> pd.DataFrame:
        """Return a monthly feature frame indexed by month-end timestamps."""

    @abstractmethod
    def compute_label(self, panel: MarketPanel) -> pd.Series:
        """Return a binary label series, indexed identically to features."""

    def assemble_dataset(self, panel: MarketPanel) -> tuple[pd.DataFrame, pd.Series]:
        features = self.compute_features(panel).dropna()
        label = self.compute_label(panel).reindex(features.index).dropna().astype(int)
        common = features.index.intersection(label.index)
        return features.loc[common], label.loc[common]


def equity_drawdown_event_label(
    panel: MarketPanel,
    drawdown_threshold: float = -0.04,
    return_threshold: float = -0.02,
    horizon_days: int = 21,
) -> pd.Series:
    """Binary label: did the next month see a meaningful equity downside?

    Mirrors the earlier baseline construction so results stay comparable
    without coupling the live framework to `build_assignment_4.py`.
    """
    equity = panel.prices["US Equity"].dropna()
    monthly_index = monthly_resample(equity).index
    labels: dict[pd.Timestamp, int] = {}
    for date in monthly_index:
        forward = equity.loc[date:].iloc[: horizon_days + 1]
        if forward.empty:
            continue
        peak = forward.cummax()
        period_dd = float((forward / peak - 1.0).min())
        period_ret = float(forward.iloc[-1] / forward.iloc[0] - 1.0)
        labels[date] = int(period_dd <= drawdown_threshold or period_ret <= return_threshold)
    return pd.Series(labels).sort_index()


def positive_growth_label(
    panel: MarketPanel,
    return_threshold: float = 0.025,
    horizon_days: int = 63,
) -> pd.Series:
    """Binary label: positive 3-month-forward equity return + low realised drawdown."""
    equity = panel.prices["US Equity"].dropna()
    monthly_index = monthly_resample(equity).index
    labels: dict[pd.Timestamp, int] = {}
    for date in monthly_index:
        forward = equity.loc[date:].iloc[: horizon_days + 1]
        if forward.empty or len(forward) < 5:
            continue
        period_ret = float(forward.iloc[-1] / forward.iloc[0] - 1.0)
        peak = forward.cummax()
        period_dd = float((forward / peak - 1.0).min())
        labels[date] = int(period_ret >= return_threshold and period_dd >= -0.05)
    return pd.Series(labels).sort_index()


def stagflation_label(
    panel: MarketPanel,
    horizon_days: int = 63,
    growth_drop_threshold: float = -0.01,
    inflation_outperf_threshold: float = 0.005,
) -> pd.Series:
    """Binary label: equity weakens AND TIPS outperforms Treasuries forward."""
    equity = panel.prices["US Equity"].dropna()
    bonds = panel.prices["US Treasuries"].dropna()
    tips = panel.tips.dropna()
    monthly_index = monthly_resample(equity).index
    labels: dict[pd.Timestamp, int] = {}
    for date in monthly_index:
        eq_fwd = equity.loc[date:].iloc[: horizon_days + 1]
        bd_fwd = bonds.loc[date:].iloc[: horizon_days + 1]
        tp_fwd = tips.loc[date:].iloc[: horizon_days + 1]
        if eq_fwd.empty or bd_fwd.empty or tp_fwd.empty or len(eq_fwd) < 5:
            continue
        eq_ret = float(eq_fwd.iloc[-1] / eq_fwd.iloc[0] - 1.0)
        tips_ret = float(tp_fwd.iloc[-1] / tp_fwd.iloc[0] - 1.0)
        treas_ret = float(bd_fwd.iloc[-1] / bd_fwd.iloc[0] - 1.0)
        labels[date] = int(eq_ret <= growth_drop_threshold and (tips_ret - treas_ret) >= inflation_outperf_threshold)
    return pd.Series(labels).sort_index()


def momentum(series: pd.Series, window_days: int) -> pd.Series:
    """Log return over a trailing calendar window."""
    return np.log(series).diff(window_days)


def realised_vol(series: pd.Series, window_days: int) -> pd.Series:
    """Annualised realised vol from daily returns."""
    rets = series.pct_change()
    return rets.rolling(window_days).std() * np.sqrt(252)


def equity_drawdown(series: pd.Series, window_days: int) -> pd.Series:
    """Trailing rolling drawdown over ``window_days``."""
    rolling_max = series.rolling(window_days, min_periods=window_days // 4).max()
    return series / rolling_max - 1.0


__all__ = [
    "Hypothesis",
    "HypothesisMeta",
    "equity_drawdown",
    "equity_drawdown_event_label",
    "momentum",
    "positive_growth_label",
    "realised_vol",
    "stagflation_label",
]
