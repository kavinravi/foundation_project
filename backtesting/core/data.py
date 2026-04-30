"""Unified data loading for prices and signals.

All loaders are repo-local and operate off the checked-in CSV cache in
`data/`. No part of the backtesting framework imports `build_assignment_4.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.core.ips import ASSET_ORDER, PRICE_TICKERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SIGNALS_DIR = DATA_DIR / "signals"

OHLCV_CANDIDATES = [
    DATA_DIR / "zion_ohlcv_daily.csv",
    DATA_DIR / "zion_ohlcv_daily_full.csv",
    DATA_DIR / "zion_ohlcv_daily_truncated.csv",
]

ROOT_ASSET_FILES = {
    "US TIPS": DATA_DIR / "0_5Y_TIPS_2002_D.csv",
}


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


def _ohlcv_path() -> Path:
    for path in OHLCV_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find a local Zion OHLCV cache. Expected one of: "
        + ", ".join(str(path) for path in OHLCV_CANDIDATES)
    )


def _load_ohlcv_table() -> pd.DataFrame:
    ohlcv = pd.read_csv(_ohlcv_path(), parse_dates=["date"])
    expected_cols = {"ticker", "date", "adj_close"}
    missing = expected_cols.difference(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV cache is missing required columns: {sorted(missing)}")
    return ohlcv


def prepare_price_panel(ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map raw Zion OHLCV rows into the base Zion price and return panel."""
    prices = (
        ohlcv.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
        .sort_index()
        .reindex(columns=PRICE_TICKERS)
    )
    prices = prices.apply(pd.to_numeric, errors="coerce")

    panel = pd.DataFrame(index=prices.index)
    panel["US Equity"] = prices["SPXT"]
    panel["US Treasuries"] = prices["LBUSTRUU"]
    panel["US REITs"] = prices["B3REITT"]
    panel["Gold"] = prices["XAU"]
    panel["Bitcoin"] = prices["XBTUSD"]
    panel = panel.sort_index()

    filled = panel.ffill()
    started = filled.notna()
    returns = filled.pct_change(fill_method=None)
    valid = started & started.shift(1, fill_value=False)
    returns = returns.where(valid)
    return filled, returns


def _load_root_asset_series() -> pd.DataFrame:
    series_map: dict[str, pd.Series] = {}
    for asset_name, path in ROOT_ASSET_FILES.items():
        series = _read_metadata_csv(path)
        series.name = asset_name
        series_map[asset_name] = series
    panel = pd.concat(series_map.values(), axis=1).sort_index() if series_map else pd.DataFrame()
    if not panel.empty:
        panel.index = pd.to_datetime(panel.index).normalize()
        panel = panel.groupby(level=0).last().sort_index()
    return panel


def _build_candidate_asset_panel(zion_prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    extra_prices = _load_root_asset_series()
    prices = pd.concat([zion_prices, extra_prices], axis=1).sort_index()
    prices = prices.groupby(level=0).last().sort_index()
    prices = prices.reindex(columns=ASSET_ORDER)

    filled = prices.ffill()
    started = filled.notna()
    returns = filled.pct_change(fill_method=None)
    valid = started & started.shift(1, fill_value=False)
    returns = returns.where(valid)
    return filled, returns


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
    """Load the candidate tradable sleeve into aligned prices and returns."""
    zion_prices, _ = prepare_price_panel(_load_ohlcv_table())
    return _build_candidate_asset_panel(zion_prices)


@lru_cache(maxsize=1)
def load_signal_panel() -> tuple[pd.DataFrame, pd.Series]:
    """Load all student-sourced signals into a single DataFrame."""
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
    "OHLCV_CANDIDATES",
    "ROOT_ASSET_FILES",
    "SIGNAL_FILES",
    "TIPS_FILE",
    "load_market_panel",
    "load_price_panel",
    "load_signal_panel",
    "monthly_resample",
    "prepare_price_panel",
    "relative_momentum",
    "trailing_window_change",
    "trend_zscore",
]
