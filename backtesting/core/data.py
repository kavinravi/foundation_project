"""Unified data loading for prices and signals.

Combines:

* the Zion 6-asset price panel (`SPXT`, `LBUSTRUU`, `B3REITT`, `XAU`,
  `XBTUSD`, `USDJPY`) loaded from the local CSV cache, and
* the cleaned student-sourced signal CSVs produced by
  ``parse_foundation_data.py`` (``data/signals/``) plus the existing
  TIPS file (``data/0_5Y_TIPS_2002_D.csv``).

Everything is exposed as time-aligned business-day frames so the
hypothesis modules can mix and match without worrying about cadence.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_assignment_4 import (  # noqa: E402
    fetch_source_tables,
    prepare_price_panel,
)

from backtesting.core.ips import ASSET_ORDER  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
SIGNALS_DIR = DATA_DIR / "signals"


@dataclass
class MarketPanel:
    """Container for time-aligned market and signal data."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    signals: pd.DataFrame
    tips: pd.Series

    def slice(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> "MarketPanel":
        sl = slice(start, end)
        return MarketPanel(
            prices=self.prices.loc[sl],
            returns=self.returns.loc[sl],
            signals=self.signals.loc[sl],
            tips=self.tips.loc[sl],
        )


def _read_metadata_csv(path: Path) -> pd.Series:
    """Read a CSV that has 5 metadata rows then a ``Date,PX_LAST`` block."""
    df = pd.read_csv(path, skiprows=5)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")["PX_LAST"].astype(float).sort_index()


SIGNAL_FILES = {
    "hy_oas": SIGNALS_DIR / "LF98OAS.csv",
    "vix": SIGNALS_DIR / "VIX.csv",
    "unemployment": SIGNALS_DIR / "Unemployment.csv",
    "umich_sentiment": SIGNALS_DIR / "UMICH_CONNSENT.csv",
    "cb_confidence": SIGNALS_DIR / "CB_CONCCONF.csv",
    "ism_pmi": SIGNALS_DIR / "ISM_Manufacturing_PMI.csv",
    "agg_oas": SIGNALS_DIR / "LBUSOAS.csv",
    "fed_funds": SIGNALS_DIR / "Fed_Funds_Rate.csv",
    "us_3m_rate": SIGNALS_DIR / "US0003M.csv",
    "hy_total_return": DATA_DIR / "LF98TRUU.csv",
    "spx_index": DATA_DIR / "SPX.csv",
    "agg_total_return": DATA_DIR / "LBUSTRUU.csv",
    "rty_index": DATA_DIR / "RTY_Index.csv",
}

TIPS_FILE = DATA_DIR / "0_5Y_TIPS_2002_D.csv"


@lru_cache(maxsize=1)
def load_price_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the 6-asset Zion price panel from the local CSV cache."""
    _, ohlcv = fetch_source_tables(refresh=False)
    filled, returns = prepare_price_panel(ohlcv)
    return filled, returns


@lru_cache(maxsize=1)
def load_signal_panel() -> tuple[pd.DataFrame, pd.Series]:
    """Load all student-sourced signals into a single DataFrame.

    Daily series are merged on the union of business-day dates and
    forward-filled (so monthly series like ISM/Confidence have a
    constant value within each month). The TIPS series is returned
    separately so hypothesis modules can compute their own derived
    spreads without polluting the signal namespace.
    """
    series_map: dict[str, pd.Series] = {}
    for name, path in SIGNAL_FILES.items():
        series = _read_metadata_csv(path)
        series.name = name
        series_map[name] = series

    tips = _read_metadata_csv(TIPS_FILE)
    tips.name = "tips_5y_total_return"

    panel = pd.concat(series_map.values(), axis=1).sort_index()
    panel = panel.reindex(columns=list(series_map.keys()))
    panel.index = pd.to_datetime(panel.index).normalize()
    panel = panel.groupby(level=0).last().sort_index()

    bd_index = pd.bdate_range(panel.index.min(), panel.index.max())
    panel = panel.reindex(bd_index).ffill()

    tips.index = pd.to_datetime(tips.index).normalize()
    tips = tips.groupby(level=0).last().sort_index().reindex(bd_index).ffill()

    return panel, tips


@lru_cache(maxsize=1)
def load_market_panel() -> MarketPanel:
    """Load prices + returns + signals + TIPS aligned on a common index."""
    prices, returns = load_price_panel()
    signals, tips = load_signal_panel()

    common_index = prices.index.union(signals.index).union(tips.index)
    common_index = common_index.sort_values()

    prices_a = prices.reindex(common_index).ffill()
    returns_a = returns.reindex(common_index)
    signals_a = signals.reindex(common_index).ffill()
    tips_a = tips.reindex(common_index).ffill()

    return MarketPanel(prices=prices_a, returns=returns_a, signals=signals_a, tips=tips_a)


def monthly_resample(series: pd.Series | pd.DataFrame, how: str = "last") -> pd.Series | pd.DataFrame:
    """Resample to month-end using the chosen aggregation."""
    return series.resample("ME").agg(how)


def trailing_window_change(series: pd.Series, window_days: int) -> pd.Series:
    """Return value minus value `window_days` ago (in calendar days)."""
    shifted = series.shift(window_days)
    return series - shifted


def relative_momentum(numerator: pd.Series, denominator: pd.Series, window_days: int) -> pd.Series:
    """Log return of (numerator / denominator) over the trailing window."""
    ratio = (numerator / denominator).replace([np.inf, -np.inf], np.nan)
    return np.log(ratio).diff(window_days)


def trend_zscore(series: pd.Series, lookback: int = 252) -> pd.Series:
    """Rolling z-score against the trailing mean / std."""
    rolling_mean = series.rolling(lookback, min_periods=lookback // 4).mean()
    rolling_std = series.rolling(lookback, min_periods=lookback // 4).std()
    return (series - rolling_mean) / rolling_std


__all__ = [
    "MarketPanel",
    "SIGNAL_FILES",
    "TIPS_FILE",
    "load_market_panel",
    "load_price_panel",
    "load_signal_panel",
    "monthly_resample",
    "relative_momentum",
    "trailing_window_change",
    "trend_zscore",
]
