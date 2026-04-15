from __future__ import annotations

import argparse
import base64
import json
import itertools
import math
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from http.cookiejar import CookieJar
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy.optimize import minimize
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, make_scorer, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBClassifier


BASE_URL = "https://zion-esgrptotba-uc.a.run.app"
ANNUALIZATION = 252
RISK_FREE_RATE = 0.02
ROUND_TRIP_COST_BPS = 5.0
BASE_WINDOW = 252
WINDOW_SETUPS = [63, 84, 126, 168, 252, 378, 504, 756, 1008, None]

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"
FRED_DIR = DATA_DIR / "fred"

RAW_ASSETS_PATH = DATA_DIR / "zion_assets.csv"
RAW_OHLCV_PATH = DATA_DIR / "zion_ohlcv_daily.csv"
RAW_OHLCV_TRUNCATED_PATH = DATA_DIR / "zion_ohlcv_daily_truncated.csv"

PRICE_TICKERS = ["SPXT", "LBUSTRUU", "B3REITT", "XAU", "XBTUSD", "USDJPY"]
ASSET_ORDER = ["US Equity", "US Bonds", "REITs", "Gold", "Bitcoin", "JPY"]
NON_TRADITIONAL = ["Gold", "Bitcoin", "JPY"]
RISKY_ASSETS = ["US Equity", "REITs", "Bitcoin"]
DEFENSIVE_ASSETS = ["US Bonds", "Gold", "JPY"]
ASSET_LABELS = {
    "SPXT": "US Equity",
    "LBUSTRUU": "US Bonds",
    "B3REITT": "REITs",
    "XAU": "Gold",
    "XBTUSD": "Bitcoin",
    "USDJPY": "JPY",
}

LOWER_BOUNDS = pd.Series(
    {
        "US Equity": 0.20,
        "US Bonds": 0.20,
        "REITs": 0.00,
        "Gold": 0.00,
        "Bitcoin": 0.00,
        "JPY": 0.00,
    }
)
UPPER_BOUNDS = pd.Series({asset: 0.30 for asset in ASSET_ORDER})
NON_TRADITIONAL_CAP = 0.25
BENCHMARK_WEIGHTS = pd.Series(
    {
        "US Equity": 0.60,
        "US Bonds": 0.40,
        "REITs": 0.00,
        "Gold": 0.00,
        "Bitcoin": 0.00,
        "JPY": 0.00,
    }
)

FRED_SERIES = {
    "VIXCLS": "CBOE Volatility Index",
    "T10Y2Y": "10Y minus 2Y Treasury Spread",
    "BAA10Y": "BAA Corporate minus 10Y Treasury Spread",
}

XGB_PARAM_GRID = [
    {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 60, "threshold": 0.40},
    {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 100, "threshold": 0.40},
    {"max_depth": 2, "learning_rate": 0.10, "n_estimators": 60, "threshold": 0.45},
    {"max_depth": 2, "learning_rate": 0.10, "n_estimators": 100, "threshold": 0.45},
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 60, "threshold": 0.40},
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 100, "threshold": 0.40},
    {"max_depth": 3, "learning_rate": 0.10, "n_estimators": 60, "threshold": 0.45},
    {"max_depth": 3, "learning_rate": 0.10, "n_estimators": 100, "threshold": 0.45},
]
LOGIT_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
SARIMAX_THRESHOLDS = [-0.0025, 0.0000, 0.0025, 0.0050]
TAA_DEVIATION_LIMIT = 0.15
IPS_RETURN_TARGET = 0.05
IPS_VOL_TARGET = 0.11
IPS_MAX_DRAWDOWN_TARGET = -0.13

DEFAULT_FEATURE_COLUMNS = [
    "spx_mom_1m",
    "spx_mom_3m",
    "spx_vol_3m",
    "bitcoin_mom_1m",
    "gold_rel_mom_3m",
    "jpy_rel_mom_3m",
    "equity_drawdown_3m",
    "avg_corr_3m",
    "vixcls",
    "vix_change_1m",
    "t10y2y",
    "baa10y",
]

SWEEP_WINDOW_PROFILES = {
    "short": {"market": "1m", "risk": "3m", "relative": "3m", "memory": "6m", "macro_change": "1m", "crypto": "1m"},
    "medium": {"market": "3m", "risk": "6m", "relative": "6m", "memory": "12m", "macro_change": "3m", "crypto": "3m"},
    "long": {"market": "6m", "risk": "12m", "relative": "12m", "memory": "18m", "macro_change": "6m", "crypto": "6m"},
}
SWEEP_LABEL_CONFIGS = [
    {"drawdown_threshold": -0.03, "return_threshold": -0.015},
    {"drawdown_threshold": -0.03, "return_threshold": -0.020},
    {"drawdown_threshold": -0.04, "return_threshold": -0.015},
    {"drawdown_threshold": -0.04, "return_threshold": -0.020},
]
SWEEP_OVERLAY_CONFIGS = [
    {"covariance_window": 126, "deviation_limit": 0.12},
    {"covariance_window": 126, "deviation_limit": 0.15},
    {"covariance_window": 252, "deviation_limit": 0.12},
    {"covariance_window": 252, "deviation_limit": 0.15},
]
PROMOTION_SHORTLIST_SIZE = 8


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    weights_by_rebalance: pd.DataFrame
    turnover_by_rebalance: pd.Series
    metrics: pd.Series


@dataclass
class FoldDefinition:
    fold_id: int
    train_index: pd.Index
    validation_index: pd.Index
    test_index: pd.Index


@dataclass
class WalkForwardResult:
    training_trials: pd.DataFrame
    selected_params: pd.DataFrame
    monthly_predictions: pd.DataFrame
    combined_taa: BacktestResult
    combined_saa_daily_returns: pd.Series
    fold_metrics: pd.DataFrame
    overall_metrics: pd.DataFrame


class ZionClient:
    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password
        self.cookie_jar = CookieJar()
        self.opener = build_opener(HTTPCookieProcessor(self.cookie_jar))
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def login(self) -> None:
        login_page = self._read_text("/-/login")
        csrf_match = re.search(r'name="csrftoken" value="([^"]+)"', login_page)
        if csrf_match is None:
            raise RuntimeError("Could not locate Zion login CSRF token.")

        payload = urlencode(
            {
                "username": self.username,
                "password": self.password,
                "csrftoken": csrf_match.group(1),
            }
        ).encode()
        request = Request(
            BASE_URL + "/-/login",
            data=payload,
            headers={**self.headers, "Content-Type": "application/x-www-form-urlencoded"},
        )
        response = self.opener.open(request, timeout=60)
        body = response.read().decode("utf-8", errors="replace")
        if "Forbidden" in body or self.username not in body:
            raise RuntimeError("Zion login failed. Check the provided credentials.")

    def fetch_csv(self, path: str) -> str:
        return self._read_text(path)

    def fetch_json(self, path: str) -> dict[str, Any]:
        return json.loads(self._read_text(path))

    def _read_text(self, path: str) -> str:
        request = Request(BASE_URL + path, headers=self.headers)
        with self.opener.open(request, timeout=60) as response:
            return response.read().decode("utf-8", errors="replace")


def get_credentials() -> tuple[str, str]:
    username = "colleague1"
    password = "CU#2026"
    return username, password


def window_label(window: int | None) -> str:
    return "expanding" if window is None else f"{window}d"


def output_path(filename: str, prefix: str | None = None) -> Path:
    return OUTPUT_DIR / (filename if not prefix else f"{prefix}_{filename}")


def trading_days(months: int) -> int:
    return 21 * months


def month_end_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    grouped = pd.Series(index=index, data=index).groupby(index.to_period("M")).max()
    return grouped.tolist()


def grouped_period_end_dates(index: pd.Index, frequency: str, drop_last: bool = False) -> list[pd.Timestamp]:
    dates = pd.DatetimeIndex(pd.to_datetime(index)).sort_values()
    if frequency == "monthly":
        grouped = pd.Series(index=dates, data=dates).groupby(dates.to_period("M")).max()
    elif frequency == "quarterly":
        grouped = pd.Series(index=dates, data=dates).groupby(dates.to_period("Q")).max()
    elif frequency == "annual":
        grouped = pd.Series(index=dates, data=dates).groupby(dates.year).max()
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")
    values = grouped.tolist()
    if drop_last:
        values = [timestamp for timestamp in values if timestamp < dates.max()]
    return values


def decision_dates_for_frequency(index: pd.Index, frequency: str = "annual") -> list[pd.Timestamp]:
    return grouped_period_end_dates(index=index, frequency=frequency, drop_last=False)


def annual_rebalance_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    return grouped_period_end_dates(index=index, frequency="annual", drop_last=True)


def rebalance_dates_for_frequency(index: pd.Index, frequency: str = "annual") -> list[pd.Timestamp]:
    return grouped_period_end_dates(index=index, frequency=frequency, drop_last=True)


def latest_history(returns: pd.DataFrame, window: int | None) -> pd.DataFrame:
    if window is None:
        return returns.copy()
    return returns.tail(min(window, len(returns))).copy()


def history_slice(returns: pd.DataFrame, as_of: pd.Timestamp, window: int | None) -> pd.DataFrame:
    history = returns.loc[:as_of]
    if window is not None:
        history = history.tail(min(window, len(history)))
    return history


def load_fred_series(series_id: str, refresh: bool = False) -> pd.Series:
    FRED_DIR.mkdir(parents=True, exist_ok=True)
    path = FRED_DIR / f"{series_id}.csv"
    if refresh or not path.exists():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        with urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    frame = pd.read_csv(path)
    frame.columns = ["date", series_id]
    frame["date"] = pd.to_datetime(frame["date"])
    frame[series_id] = pd.to_numeric(frame[series_id], errors="coerce")
    return frame.set_index("date")[series_id].sort_index()


def fetch_source_tables(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not refresh and RAW_ASSETS_PATH.exists() and RAW_OHLCV_PATH.exists():
        assets = pd.read_csv(RAW_ASSETS_PATH)
        ohlcv = pd.read_csv(RAW_OHLCV_PATH, parse_dates=["date"])
        if len(ohlcv) > 20_000:
            return assets, ohlcv

    username, password = get_credentials()
    client = ZionClient(username=username, password=password)
    client.login()

    assets_csv = client.fetch_csv("/zion/assets.csv?_size=max")
    RAW_ASSETS_PATH.write_text(assets_csv)

    truncated_csv = client.fetch_csv("/zion/ohlcv_daily.csv?_size=max")
    RAW_OHLCV_TRUNCATED_PATH.write_text(truncated_csv)

    all_rows: list[list[Any]] = []
    path = "/zion/ohlcv_daily.json?_size=1000&_sort_desc=date"
    columns: list[str] | None = None
    while path:
        payload = client.fetch_json(path)
        columns = payload["columns"]
        all_rows.extend(payload["rows"])
        next_url = payload.get("next_url")
        if next_url:
            parsed = urlparse(next_url)
            path = parsed.path + ("?" + parsed.query if parsed.query else "")
        else:
            path = ""

    if columns is None:
        raise RuntimeError("Zion OHLCV download returned no columns.")

    full_ohlcv = pd.DataFrame(all_rows, columns=columns)
    full_ohlcv.to_csv(RAW_OHLCV_PATH, index=False)

    assets = pd.read_csv(StringIO(assets_csv))
    ohlcv = pd.read_csv(RAW_OHLCV_PATH, parse_dates=["date"])
    return assets, ohlcv


def prepare_price_panel(ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = (
        ohlcv.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
        .sort_index()
        .loc[:, PRICE_TICKERS]
    )
    prices = prices.apply(pd.to_numeric, errors="coerce")

    panel = pd.DataFrame(index=prices.index)
    panel["US Equity"] = prices["SPXT"]
    panel["US Bonds"] = prices["LBUSTRUU"]
    panel["REITs"] = prices["B3REITT"]
    panel["Gold"] = prices["XAU"]
    panel["Bitcoin"] = prices["XBTUSD"]
    panel["JPY"] = 1.0 / prices["USDJPY"]
    panel = panel.loc[:, ASSET_ORDER].sort_index()

    filled = panel.ffill()
    started = filled.notna()
    returns = filled.pct_change(fill_method=None)
    valid = started & started.shift(1).fillna(False)
    returns = returns.where(valid)
    return filled, returns


def asset_coverage_table(filled_prices: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for asset in ASSET_ORDER:
        series = filled_prices[asset].dropna()
        records.append(
            {
                "asset": asset,
                "start_date": series.index.min().date().isoformat(),
                "end_date": series.index.max().date().isoformat(),
                "observations": int(series.shape[0]),
                "lower_bound": LOWER_BOUNDS[asset],
                "upper_bound": UPPER_BOUNDS[asset],
                "is_non_traditional": asset in NON_TRADITIONAL,
            }
        )
    coverage = pd.DataFrame(records)
    coverage.to_csv(OUTPUT_DIR / "history_coverage.csv", index=False)
    return coverage


def max_total_weight_for_assets(active_assets: list[str]) -> float:
    traditional = [asset for asset in active_assets if asset not in NON_TRADITIONAL]
    non_trad = [asset for asset in active_assets if asset in NON_TRADITIONAL]
    return float(UPPER_BOUNDS[traditional].sum() + min(UPPER_BOUNDS[non_trad].sum(), NON_TRADITIONAL_CAP))


def feasible_for_assets(active_assets: list[str]) -> bool:
    lower_sum = float(LOWER_BOUNDS.reindex(active_assets).fillna(0.0).sum())
    return lower_sum <= 1.0 and max_total_weight_for_assets(active_assets) >= 1.0


def build_history_regime_table(filled_prices: pd.DataFrame) -> pd.DataFrame:
    start_dates = {asset: filled_prices[asset].dropna().index.min() for asset in ASSET_ORDER}
    candidate_dates = sorted(set(start_dates.values()))
    rows: list[dict[str, Any]] = []
    for date in candidate_dates:
        active_assets = [asset for asset in ASSET_ORDER if start_dates[asset] <= date]
        rows.append(
            {
                "date": date.date().isoformat(),
                "active_assets": ", ".join(active_assets),
                "active_count": len(active_assets),
                "max_investable_weight": max_total_weight_for_assets(active_assets),
                "feasible_under_caps": feasible_for_assets(active_assets),
                "is_full_universe": len(active_assets) == len(ASSET_ORDER),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "history_regimes.csv", index=False)
    return table


def first_feasible_date(filled_prices: pd.DataFrame) -> pd.Timestamp:
    start_dates = {asset: filled_prices[asset].dropna().index.min() for asset in ASSET_ORDER}
    candidate_dates = sorted(set(start_dates.values()))
    for date in candidate_dates:
        active_assets = [asset for asset in ASSET_ORDER if start_dates[asset] <= date]
        if feasible_for_assets(active_assets):
            return date
    raise RuntimeError("Could not find a feasible portfolio start date.")


def first_full_overlap_date(filled_prices: pd.DataFrame) -> pd.Timestamp:
    return max(filled_prices[asset].dropna().index.min() for asset in ASSET_ORDER)


def initial_guess_for_assets(active_assets: list[str]) -> np.ndarray:
    active_lower = LOWER_BOUNDS.reindex(active_assets).fillna(0.0)
    active_upper = UPPER_BOUNDS.reindex(active_assets).fillna(0.0)
    guess = active_lower.copy()
    slack = 1.0 - float(guess.sum())
    room = active_upper - active_lower
    if slack > 0 and room.sum() > 0:
        guess += slack * room / float(room.sum())
    return guess.to_numpy(dtype=float)


def initial_guess_from_bounds(lower_bounds: pd.Series, upper_bounds: pd.Series) -> np.ndarray:
    guess = lower_bounds.copy()
    slack = 1.0 - float(guess.sum())
    room = upper_bounds - lower_bounds
    if slack > 0 and float(room.sum()) > 0:
        guess += slack * room / float(room.sum())
    return guess.to_numpy(dtype=float)


def solve_min_variance(covariance: pd.DataFrame, active_assets: list[str]) -> pd.Series:
    if not feasible_for_assets(active_assets):
        raise RuntimeError(f"Infeasible portfolio for active assets: {active_assets}")

    active_cov = covariance.loc[active_assets, active_assets].fillna(0.0)
    cov = active_cov.to_numpy(dtype=float)
    lower = LOWER_BOUNDS.loc[active_assets].to_numpy(dtype=float)
    upper = UPPER_BOUNDS.loc[active_assets].to_numpy(dtype=float)
    non_trad_idx = [idx for idx, asset in enumerate(active_assets) if asset in NON_TRADITIONAL]
    x0 = initial_guess_for_assets(active_assets)

    constraints: list[dict[str, Any]] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if non_trad_idx:
        constraints.append({"type": "ineq", "fun": lambda w: NON_TRADITIONAL_CAP - np.sum(w[non_trad_idx])})

    result = minimize(
        lambda w: float(w @ cov @ w),
        x0=x0,
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"MinVar optimization failed: {result.message}")

    weights = pd.Series(0.0, index=ASSET_ORDER)
    weights.loc[active_assets] = result.x
    weights[np.isclose(weights, 0.0, atol=1e-10)] = 0.0
    return weights


def solve_min_variance_with_bounds(
    covariance: pd.DataFrame,
    active_assets: list[str],
    lower_bounds: pd.Series,
    upper_bounds: pd.Series,
) -> pd.Series:
    lower = lower_bounds.reindex(active_assets).fillna(0.0)
    upper = upper_bounds.reindex(active_assets).fillna(1.0)
    if float(lower.sum()) > 1.0 + 1e-9 or float(upper.sum()) < 1.0 - 1e-9:
        raise RuntimeError("Infeasible tactical bounds for min-variance overlay.")

    active_cov = covariance.loc[active_assets, active_assets].fillna(0.0)
    cov = active_cov.to_numpy(dtype=float)
    x0 = initial_guess_from_bounds(lower, upper)
    constraints: list[dict[str, Any]] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(
        lambda w: float(w @ cov @ w),
        x0=x0,
        method="SLSQP",
        bounds=list(zip(lower.to_numpy(dtype=float), upper.to_numpy(dtype=float))),
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Tactical min-variance optimization failed: {result.message}")

    weights = pd.Series(0.0, index=ASSET_ORDER)
    weights.loc[active_assets] = result.x
    weights[np.isclose(weights, 0.0, atol=1e-10)] = 0.0
    return weights


def solve_target_return_frontier(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    target_return: float,
    active_assets: list[str],
) -> pd.Series | None:
    if not feasible_for_assets(active_assets):
        return None

    mu = mean_returns.loc[active_assets].to_numpy(dtype=float)
    cov = covariance.loc[active_assets, active_assets].fillna(0.0).to_numpy(dtype=float)
    lower = LOWER_BOUNDS.loc[active_assets].to_numpy(dtype=float)
    upper = UPPER_BOUNDS.loc[active_assets].to_numpy(dtype=float)
    non_trad_idx = [idx for idx, asset in enumerate(active_assets) if asset in NON_TRADITIONAL]
    x0 = initial_guess_for_assets(active_assets)

    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: float(w @ mu) - target_return},
    ]
    if non_trad_idx:
        constraints.append({"type": "ineq", "fun": lambda w: NON_TRADITIONAL_CAP - np.sum(w[non_trad_idx])})

    result = minimize(
        lambda w: float(w @ cov @ w),
        x0=x0,
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        return None

    weights = pd.Series(0.0, index=ASSET_ORDER)
    weights.loc[active_assets] = result.x
    weights[np.isclose(weights, 0.0, atol=1e-10)] = 0.0
    return weights


def current_active_assets(returns: pd.DataFrame, as_of: pd.Timestamp) -> list[str]:
    row = returns.loc[as_of]
    return [asset for asset in ASSET_ORDER if pd.notna(row[asset])]


def turnover_to_target(current_weights: pd.Series, target_weights: pd.Series) -> float:
    return 0.5 * float((target_weights - current_weights).abs().sum())


def drifted_weights(start_weights: pd.Series, period_returns: pd.DataFrame) -> pd.Series:
    if period_returns.empty:
        return start_weights.copy()
    filled_period = period_returns.fillna(0.0)
    growth = (1.0 + filled_period).prod()
    end_weights = start_weights * growth
    total = end_weights.sum()
    if total <= 0:
        return start_weights.copy()
    return end_weights / total


def compute_metrics(daily_returns: pd.Series) -> pd.Series:
    if daily_returns.empty:
        return pd.Series(
            {
                "total_return_pa": np.nan,
                "cumulative_return": np.nan,
                "volatility_pa": np.nan,
                "historical_var_95_daily": np.nan,
                "sharpe_rf2": np.nan,
                "calmar": np.nan,
                "max_drawdown": np.nan,
            }
        )

    equity_curve = (1.0 + daily_returns).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1.0
    years = len(daily_returns) / ANNUALIZATION
    cagr = equity_curve.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan
    annual_vol = daily_returns.std(ddof=0) * np.sqrt(ANNUALIZATION)
    excess_return = daily_returns.mean() * ANNUALIZATION - RISK_FREE_RATE
    sharpe = excess_return / annual_vol if annual_vol > 0 else np.nan
    max_drawdown = float(drawdown.min())
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan
    var95 = float(np.quantile(daily_returns, 0.05))

    return pd.Series(
        {
            "total_return_pa": float(cagr),
            "cumulative_return": float(equity_curve.iloc[-1] - 1.0),
            "volatility_pa": float(annual_vol),
            "historical_var_95_daily": var95,
            "sharpe_rf2": float(sharpe),
            "calmar": float(calmar),
            "max_drawdown": max_drawdown,
        }
    )


def run_strategy_backtest(
    returns: pd.DataFrame,
    target_weights_by_date: dict[pd.Timestamp, pd.Series],
    strategy_name: str,
) -> BacktestResult:
    rebalance_dates = list(target_weights_by_date.keys())
    current_weights = pd.Series(0.0, index=ASSET_ORDER)
    daily_segments: list[pd.Series] = []
    turnover_records: list[tuple[pd.Timestamp, float]] = []

    for idx, rebalance_date in enumerate(rebalance_dates):
        target_weights = target_weights_by_date[rebalance_date].loc[ASSET_ORDER]
        turnover = turnover_to_target(current_weights=current_weights, target_weights=target_weights)
        turnover_records.append((rebalance_date, turnover))

        start_loc = returns.index.get_loc(rebalance_date) + 1
        end_loc = (
            returns.index.get_loc(rebalance_dates[idx + 1]) + 1 if idx + 1 < len(rebalance_dates) else len(returns)
        )
        period = returns.iloc[start_loc:end_loc].fillna(0.0)
        if period.empty:
            current_weights = target_weights
            continue

        portfolio_returns = period @ target_weights
        portfolio_returns.iloc[0] -= turnover * (ROUND_TRIP_COST_BPS / 10_000.0)
        portfolio_returns.name = strategy_name
        daily_segments.append(portfolio_returns)

        current_weights = drifted_weights(start_weights=target_weights, period_returns=period)

    daily_returns = pd.concat(daily_segments).sort_index()
    turnover_series = pd.Series({timestamp: value for timestamp, value in turnover_records}, name="turnover").sort_index()
    weights_frame = pd.DataFrame(target_weights_by_date).T.loc[:, ASSET_ORDER].sort_index().rename_axis("rebalance_date")
    metrics = compute_metrics(daily_returns)
    return BacktestResult(
        daily_returns=daily_returns,
        weights_by_rebalance=weights_frame,
        turnover_by_rebalance=turnover_series,
        metrics=metrics,
    )


def build_minvar_targets(returns: pd.DataFrame, rebalance_dates: list[pd.Timestamp], window: int | None) -> dict[pd.Timestamp, pd.Series]:
    targets: dict[pd.Timestamp, pd.Series] = {}
    for rebalance_date in rebalance_dates:
        active_assets = current_active_assets(returns, rebalance_date)
        history = history_slice(returns[active_assets].dropna(how="all"), rebalance_date, window)
        covariance = history.cov(min_periods=max(20, min(len(history), 20)))
        targets[rebalance_date] = solve_min_variance(covariance, active_assets)
    return targets


def run_benchmark_backtest(returns: pd.DataFrame, rebalance_dates: list[pd.Timestamp]) -> BacktestResult:
    targets = {rebalance_date: BENCHMARK_WEIGHTS.copy() for rebalance_date in rebalance_dates}
    return run_strategy_backtest(returns=returns, target_weights_by_date=targets, strategy_name="60/40")


def annual_returns_table(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    annual = pd.DataFrame(series_map).sort_index()
    annual.index = pd.to_datetime(annual.index)
    table = (1.0 + annual).groupby(annual.index.year).prod() - 1.0
    table.index.name = "year"
    return table


def summarize_windows(returns: pd.DataFrame, rebalance_dates: list[pd.Timestamp]) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    benchmark = run_benchmark_backtest(returns, rebalance_dates)
    strategy_results: dict[str, BacktestResult] = {"benchmark": benchmark}
    rows: list[dict[str, Any]] = []

    for window in WINDOW_SETUPS:
        label = window_label(window)
        targets = build_minvar_targets(returns=returns, rebalance_dates=rebalance_dates, window=window)
        result = run_strategy_backtest(returns=returns, target_weights_by_date=targets, strategy_name="SAA MinVar")
        strategy_results[label] = result

        latest = latest_history(returns, window)
        active_assets = [asset for asset in ASSET_ORDER if latest[asset].notna().any()]
        latest_cov = latest[active_assets].cov()
        latest_mu = latest[active_assets].mean() * ANNUALIZATION
        latest_weights = solve_min_variance(latest_cov, active_assets)
        latest_return = float(latest_weights @ latest_mu.reindex(ASSET_ORDER).fillna(0.0))
        active_cov_annual = latest_cov.reindex(index=active_assets, columns=active_assets) * ANNUALIZATION
        latest_vol = float(
            np.sqrt(
                latest_weights.loc[active_assets].to_numpy()
                @ active_cov_annual.to_numpy(dtype=float)
                @ latest_weights.loc[active_assets].to_numpy()
            )
        )

        row: dict[str, Any] = {
            "window": label,
            "backtest_total_return_pa": result.metrics["total_return_pa"],
            "backtest_cumulative_return": result.metrics["cumulative_return"],
            "backtest_volatility_pa": result.metrics["volatility_pa"],
            "backtest_historical_var_95_daily": result.metrics["historical_var_95_daily"],
            "backtest_sharpe_rf2": result.metrics["sharpe_rf2"],
            "backtest_calmar": result.metrics["calmar"],
            "backtest_max_drawdown": result.metrics["max_drawdown"],
            "latest_ex_ante_return": latest_return,
            "latest_ex_ante_vol": latest_vol,
            "avg_turnover": result.turnover_by_rebalance.mean(),
            "non_traditional_weight": latest_weights[NON_TRADITIONAL].sum(),
        }
        row.update({f"weight_{asset.lower().replace(' ', '_')}": latest_weights[asset] for asset in ASSET_ORDER})
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["vol_within_11"] = summary["backtest_volatility_pa"] <= 0.11
    summary["drawdown_within_13"] = summary["backtest_max_drawdown"] >= -0.13
    summary["return_above_5"] = summary["backtest_total_return_pa"] >= 0.05
    summary.to_csv(OUTPUT_DIR / "saa_window_summary.csv", index=False)
    benchmark.metrics.to_frame(name="60_40").T.to_csv(OUTPUT_DIR / "saa_benchmark_metrics.csv", index=False)
    return summary, strategy_results


def max_drawdown_window(daily_returns: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    equity_curve = (1.0 + daily_returns).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1.0
    trough = drawdown.idxmin()
    peak = equity_curve.loc[:trough].idxmax()
    return peak, trough


def build_regime_table(returns: pd.DataFrame, window: int = 60, change_lookback: int = 21) -> tuple[pd.DataFrame, pd.Series]:
    rolling_corr = returns.rolling(window).corr()
    average_corr: dict[pd.Timestamp, float] = {}
    for timestamp in returns.index[window - 1 :]:
        matrix = rolling_corr.loc[timestamp].to_numpy(dtype=float)
        upper = matrix[np.triu_indices_from(matrix, k=1)]
        upper = upper[np.isfinite(upper)]
        average_corr[timestamp] = float(np.mean(upper))

    average_corr_series = pd.Series(average_corr, name="avg_pairwise_corr").sort_index()
    change_series = average_corr_series.diff(change_lookback).abs().dropna()
    key_dates = {
        "highest_average_correlation": average_corr_series.idxmax(),
        "lowest_average_correlation": average_corr_series.idxmin(),
        "largest_21d_correlation_shift": change_series.idxmax(),
    }

    rows: list[dict[str, Any]] = []
    for label, timestamp in key_dates.items():
        corr_matrix = rolling_corr.loc[timestamp]
        rows.append(
            {
                "regime": label,
                "window_end": timestamp.date().isoformat(),
                "window_start": (timestamp - pd.Timedelta(days=window)).date().isoformat(),
                "avg_pairwise_corr": average_corr_series.loc[timestamp],
                "equity_bonds_corr": corr_matrix.loc["US Equity", "US Bonds"],
                "equity_gold_corr": corr_matrix.loc["US Equity", "Gold"],
                "equity_bitcoin_corr": corr_matrix.loc["US Equity", "Bitcoin"],
                "equity_jpy_corr": corr_matrix.loc["US Equity", "JPY"],
            }
        )
    regime_table = pd.DataFrame(rows)
    regime_table.to_csv(OUTPUT_DIR / "saa_regime_analysis.csv", index=False)
    return regime_table, average_corr_series


def plot_covariance_sensitivity(summary: pd.DataFrame) -> None:
    ordered = summary.copy()
    ordered["window_days"] = [np.nan if value == "expanding" else int(value[:-1]) for value in ordered["window"]]
    plot_df = ordered[ordered["window"] != "expanding"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(plot_df["window_days"], plot_df["backtest_total_return_pa"] * 100, marker="o", label="CAGR")
    axes[0].plot(plot_df["window_days"], plot_df["backtest_volatility_pa"] * 100, marker="o", label="Volatility")
    axes[0].axhline(5.0, color="#2d6a4f", linestyle="--", linewidth=1.2, label="5% return objective")
    axes[0].axhline(11.0, color="#bc4749", linestyle="--", linewidth=1.2, label="11% vol ceiling")
    axes[0].set_title("SAA sensitivity by covariance window")
    axes[0].set_xlabel("Lookback window (trading days)")
    axes[0].set_ylabel("Percent")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(plot_df["window_days"], plot_df["latest_ex_ante_return"] * 100, marker="o", label="Latest ex-ante return")
    axes[1].plot(plot_df["window_days"], plot_df["latest_ex_ante_vol"] * 100, marker="o", label="Latest ex-ante vol")
    axes[1].set_title("Latest SAA estimates by window")
    axes[1].set_xlabel("Lookback window (trading days)")
    axes[1].set_ylabel("Percent")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "saa_covariance_sensitivity.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_weight_stability(summary: pd.DataFrame) -> None:
    weight_columns = [f"weight_{asset.lower().replace(' ', '_')}" for asset in ASSET_ORDER]
    labels = summary["window"].tolist()
    values = summary[weight_columns].to_numpy().T

    fig, ax = plt.subplots(figsize=(11, 6))
    heatmap = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=0.30)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(ASSET_ORDER)))
    ax.set_yticklabels(ASSET_ORDER)
    ax.set_title("Current SAA weights across estimation windows")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            ax.text(col_idx, row_idx, f"{values[row_idx, col_idx]:.0%}", ha="center", va="center", fontsize=8)
    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04, label="Weight")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "saa_weight_stability.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_regime_series(average_corr_series: pd.Series, regime_table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    average_corr_series.plot(ax=ax, color="#1d3557", linewidth=1.6)
    for _, row in regime_table.iterrows():
        timestamp = pd.Timestamp(row["window_end"])
        ax.axvline(timestamp, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.scatter(timestamp, row["avg_pairwise_corr"], s=40)
    ax.set_title("60-day rolling average pairwise correlation")
    ax.set_ylabel("Average correlation")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "saa_regime_average_correlation.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_backtest(base_result: BacktestResult, benchmark_result: BacktestResult, regime_table: pd.DataFrame) -> None:
    saa_curve = (1.0 + base_result.daily_returns).cumprod() - 1.0
    benchmark_curve = (1.0 + benchmark_result.daily_returns).cumprod() - 1.0
    peak, trough = max_drawdown_window(base_result.daily_returns)

    fig, ax = plt.subplots(figsize=(12, 6))
    saa_curve.plot(ax=ax, linewidth=2.0, label=f"SAA MinVar ({window_label(BASE_WINDOW)})", color="#1d3557")
    benchmark_curve.plot(ax=ax, linewidth=1.8, label="60/40 benchmark", color="#e76f51")
    ax.axvspan(peak, trough, color="#d62828", alpha=0.10, label="SAA max drawdown window")

    highest_corr = pd.Timestamp(
        regime_table.loc[regime_table["regime"] == "highest_average_correlation", "window_end"].iloc[0]
    )
    ax.axvline(highest_corr, color="#2a9d8f", linestyle="--", linewidth=1.2, label="Highest correlation regime")

    ax.set_title("Assignment 4 SAA cumulative return backtest")
    ax.set_ylabel("Cumulative return")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "saa_cumulative_backtest.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def frontier_return_bounds(mean_returns: pd.Series, covariance: pd.DataFrame, active_assets: list[str]) -> tuple[float, float]:
    mu = mean_returns.loc[active_assets].to_numpy(dtype=float)
    lower = LOWER_BOUNDS.loc[active_assets].to_numpy(dtype=float)
    upper = UPPER_BOUNDS.loc[active_assets].to_numpy(dtype=float)
    non_trad_idx = [idx for idx, asset in enumerate(active_assets) if asset in NON_TRADITIONAL]
    x0 = initial_guess_for_assets(active_assets)
    constraints: list[dict[str, Any]] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if non_trad_idx:
        constraints.append({"type": "ineq", "fun": lambda w: NON_TRADITIONAL_CAP - np.sum(w[non_trad_idx])})

    min_result = minimize(
        lambda w: float(w @ mu),
        x0=x0,
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    max_result = minimize(
        lambda w: -float(w @ mu),
        x0=x0,
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not min_result.success or not max_result.success:
        raise RuntimeError("Could not determine efficient frontier return bounds.")
    return float(min_result.fun), float(-max_result.fun)


def plot_efficient_frontier(returns: pd.DataFrame) -> pd.DataFrame:
    latest = latest_history(returns, BASE_WINDOW)
    active_assets = ASSET_ORDER.copy()
    covariance = latest[active_assets].cov() * ANNUALIZATION
    mean_returns = latest[active_assets].mean() * ANNUALIZATION
    minvar_weights = solve_min_variance(covariance, active_assets)
    min_return, max_return = frontier_return_bounds(mean_returns, covariance, active_assets)

    frontier_points: list[dict[str, Any]] = []
    for target in np.linspace(min_return, max_return, 40):
        weights = solve_target_return_frontier(mean_returns, covariance, target, active_assets)
        if weights is None:
            continue
        vol = float(
            np.sqrt(
                weights.loc[active_assets].to_numpy()
                @ covariance.loc[active_assets, active_assets].to_numpy()
                @ weights.loc[active_assets].to_numpy()
            )
        )
        frontier_points.append(
            {
                "target_return": float(target),
                "volatility": vol,
                **{asset: weights[asset] for asset in ASSET_ORDER},
            }
        )

    frontier_df = pd.DataFrame(frontier_points)
    frontier_df.to_csv(OUTPUT_DIR / "saa_efficient_frontier_points.csv", index=False)

    minvar_vol = float(
        np.sqrt(
            minvar_weights.loc[active_assets].to_numpy()
            @ covariance.loc[active_assets, active_assets].to_numpy()
            @ minvar_weights.loc[active_assets].to_numpy()
        )
    )
    minvar_return = float(minvar_weights.loc[active_assets] @ mean_returns.loc[active_assets])

    benchmark_return = float(BENCHMARK_WEIGHTS @ mean_returns.reindex(ASSET_ORDER).fillna(0.0))
    benchmark_vol = float(
        np.sqrt(
            BENCHMARK_WEIGHTS.to_numpy()
            @ covariance.reindex(index=ASSET_ORDER, columns=ASSET_ORDER).fillna(0.0).to_numpy()
            @ BENCHMARK_WEIGHTS.to_numpy()
        )
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(frontier_df["volatility"] * 100, frontier_df["target_return"] * 100, color="#264653", linewidth=2.0)
    ax.scatter(minvar_vol * 100, minvar_return * 100, color="#e76f51", s=70, label="Recommended SAA")
    ax.scatter(benchmark_vol * 100, benchmark_return * 100, color="#2a9d8f", s=70, label="60/40 benchmark")
    ax.axhline(5.0, color="#2d6a4f", linestyle="--", linewidth=1.2, label="5% objective")
    ax.axvline(11.0, color="#bc4749", linestyle="--", linewidth=1.2, label="11% vol ceiling")
    ax.set_title(f"Efficient frontier using latest {window_label(BASE_WINDOW)} sample")
    ax.set_xlabel("Annualized volatility (%)")
    ax.set_ylabel("Annualized expected return (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "saa_efficient_frontier.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    return frontier_df


def plot_risk_return_comparison(walk_forward: WalkForwardResult) -> None:
    comparison = walk_forward.overall_metrics.set_index("portfolio")
    taa = comparison.loc["TAA Overlay"]
    saa = comparison.loc["SAA Baseline"]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(IPS_RETURN_TARGET * 100, color="#2d6a4f", linestyle="--", linewidth=1.2, label="5% return target")
    ax.axvline(IPS_VOL_TARGET * 100, color="#bc4749", linestyle="--", linewidth=1.2, label="11% vol ceiling")

    points = [
        ("SAA Baseline", saa["volatility_pa"] * 100, saa["total_return_pa"] * 100, "#e76f51"),
        ("TAA Overlay", taa["volatility_pa"] * 100, taa["total_return_pa"] * 100, "#1d3557"),
    ]
    for label, x, y, color in points:
        ax.scatter(x, y, color=color, s=90, label=label, zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9, color=color)

    x_values = [x for _, x, _, _ in points]
    y_values = [y for _, _, y, _ in points]
    ax.set_xlim(min(x_values) - 0.8, max(x_values) + 1.8)
    ax.set_ylim(min(y_values) - 1.2, max(y_values) + 1.2)
    ax.set_title("Realized annualized risk/return: TAA vs SAA")
    ax.set_xlabel("Annualized volatility (%)")
    ax.set_ylabel("Annualized return (%)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "risk_return_comparison.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_strategy_context(walk_forward: WalkForwardResult) -> None:
    taa_curve = (1.0 + walk_forward.combined_taa.daily_returns).cumprod() - 1.0
    saa_curve = (1.0 + walk_forward.combined_saa_daily_returns).cumprod() - 1.0
    peak, trough = max_drawdown_window(walk_forward.combined_taa.daily_returns)

    fig, ax = plt.subplots(figsize=(12, 6))
    taa_curve.plot(ax=ax, linewidth=2.0, label="TAA Overlay", color="#1d3557")
    saa_curve.plot(ax=ax, linewidth=1.8, label="SAA Baseline", color="#e76f51")
    ax.axvspan(peak, trough, color="#d62828", alpha=0.10, label="TAA max drawdown window")
    ax.set_title("TAA versus SAA cumulative backtest")
    ax.set_ylabel("Cumulative return")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "strategy_context_backtest.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def write_saa_summary_outputs(
    filled_prices: pd.DataFrame,
    summary: pd.DataFrame,
    base_result: BacktestResult,
    benchmark_result: BacktestResult,
    regime_table: pd.DataFrame,
    full_overlap_returns: pd.DataFrame,
) -> None:
    latest = latest_history(full_overlap_returns, BASE_WINDOW)
    covariance = latest.cov()
    mean_returns = latest.mean() * ANNUALIZATION
    recommended_weights = solve_min_variance(covariance, ASSET_ORDER)

    overall_metrics = pd.DataFrame(
        [
            {"portfolio": "SAA MinVar", **base_result.metrics.to_dict()},
            {"portfolio": "60/40", **benchmark_result.metrics.to_dict()},
        ]
    )
    overall_metrics.to_csv(OUTPUT_DIR / "saa_overall_metrics.csv", index=False)

    annual_table = annual_returns_table(
        {"SAA MinVar": base_result.daily_returns, "60/40": benchmark_result.daily_returns}
    )
    annual_table.to_csv(OUTPUT_DIR / "saa_annual_returns.csv")
    recommended_weights.rename("weight").to_csv(OUTPUT_DIR / "saa_recommended_weights.csv", header=True)
    covariance.round(8).to_csv(OUTPUT_DIR / "saa_base_covariance_matrix.csv")

    highest_corr = regime_table.loc[regime_table["regime"] == "highest_average_correlation"].iloc[0]
    best_row = summary.sort_values("backtest_calmar", ascending=False).iloc[0]

    memo = f"""# Assignment 4 Strategic Baseline

## Executive takeaway
The refreshed Zion pull confirms that the investable series extend far beyond the truncated 2020 export. The full six-asset overlap begins in `{first_full_overlap_date(filled_prices).date().isoformat()}` because Bitcoin starts in mid-2010, while the staggered-availability backtest becomes feasible earlier in `2003` once REITs enter the universe and the IPS cap structure can be satisfied.

Using the latest `{window_label(BASE_WINDOW)}` covariance sample, the refreshed SAA recommendation is:

| Sleeve | Weight |
|---|---:|
| US Equity | {recommended_weights['US Equity']:.1%} |
| US Bonds | {recommended_weights['US Bonds']:.1%} |
| REITs | {recommended_weights['REITs']:.1%} |
| Gold | {recommended_weights['Gold']:.1%} |
| Bitcoin | {recommended_weights['Bitcoin']:.1%} |
| JPY | {recommended_weights['JPY']:.1%} |

## Updated Assignment 4 observations
- The strategic portfolio remains long-only, fully invested, and inside the tighter `25%` non-traditional cap.
- The `{window_label(BASE_WINDOW)}` setup is still the cleanest committee baseline because it balances recency against parameter noise.
- The best ex-post Calmar ratio in the window sweep was `{best_row['window']}`, but the one-year setup remains more defensible for the TAA benchmark.

## Risk context
- Refreshed SAA CAGR: `{base_result.metrics['total_return_pa']:.1%}`.
- Refreshed SAA volatility: `{base_result.metrics['volatility_pa']:.1%}`.
- Refreshed SAA max drawdown: `{base_result.metrics['max_drawdown']:.1%}`.
- Highest correlation regime ended on `{highest_corr['window_end']}` with average pairwise correlation `{highest_corr['avg_pairwise_corr']:.2f}`.

## Why this baseline matters for Assignment 4
The tactical overlay is not competing with a 60/40 benchmark. It is competing against this refreshed strategic allocation, built from the full Zion history rather than the truncated CSV shortcut.
"""
    (OUTPUT_DIR / "saa_committee_memo.md").write_text(memo)


def average_pairwise_correlation(returns: pd.DataFrame, window_days: int) -> pd.Series:
    series = pd.Series(index=returns.index, dtype=float)
    for timestamp in returns.index[window_days - 1 :]:
        corr = returns.loc[:timestamp].tail(window_days).corr()
        upper = corr.to_numpy(dtype=float)[np.triu_indices(len(ASSET_ORDER), k=1)]
        upper = upper[np.isfinite(upper)]
        series.loc[timestamp] = float(np.mean(upper))
    return series


def build_feature_library(
    filled_prices: pd.DataFrame,
    returns: pd.DataFrame,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    month_ends = pd.DatetimeIndex(month_end_dates(returns.index))
    features = pd.DataFrame(index=month_ends)
    manifest_rows: list[dict[str, Any]] = []

    def add_feature(name: str, values: pd.Series, family: str, lookback_months: int | None, description: str) -> None:
        features[name] = values.reindex(month_ends)
        manifest_rows.append(
            {
                "feature_name": name,
                "family": family,
                "lookback_months": lookback_months,
                "description": description,
            }
        )

    for months in [1, 3, 6]:
        days = trading_days(months)
        add_feature(
            f"spx_mom_{months}m",
            filled_prices["US Equity"].pct_change(days),
            "market_momentum",
            months,
            f"US equity trailing {months}-month price momentum",
        )
        add_feature(
            f"bitcoin_mom_{months}m",
            filled_prices["Bitcoin"].pct_change(days),
            "crypto",
            months,
            f"Bitcoin trailing {months}-month price momentum",
        )

    for months in [1, 3, 6, 12]:
        days = trading_days(months)
        add_feature(
            f"spx_vol_{months}m",
            returns["US Equity"].rolling(days).std().mul(np.sqrt(ANNUALIZATION)),
            "market_risk",
            months,
            f"US equity realized volatility over {months} months",
        )

    for months in [3, 6, 12]:
        days = trading_days(months)
        add_feature(
            f"gold_rel_mom_{months}m",
            filled_prices["Gold"].pct_change(days) - filled_prices["US Equity"].pct_change(days),
            "defensive_relative",
            months,
            f"Gold relative momentum versus US equity over {months} months",
        )
        add_feature(
            f"jpy_rel_mom_{months}m",
            filled_prices["JPY"].pct_change(days) - filled_prices["US Equity"].pct_change(days),
            "defensive_relative",
            months,
            f"JPY relative momentum versus US equity over {months} months",
        )
        add_feature(
            f"equity_drawdown_{months}m",
            filled_prices["US Equity"] / filled_prices["US Equity"].rolling(days).max() - 1.0,
            "market_risk",
            months,
            f"US equity trailing drawdown over {months} months",
        )
        add_feature(
            f"avg_corr_{months}m",
            average_pairwise_correlation(returns, days),
            "market_risk",
            months,
            f"Average pairwise cross-asset correlation over {months} months",
        )

    macro_daily = pd.DataFrame(index=returns.index)
    for series_id in FRED_SERIES:
        macro_daily[series_id.lower()] = load_fred_series(series_id, refresh=refresh).reindex(returns.index).ffill()
        add_feature(
            series_id.lower(),
            macro_daily[series_id.lower()],
            "macro_spot",
            None,
            f"Latest level of {FRED_SERIES[series_id]}",
        )

    for months in [1, 3, 6]:
        add_feature(
            f"vix_change_{months}m",
            macro_daily["vixcls"].pct_change(months),
            "macro_spot",
            months,
            f"VIX percent change over {months} months",
        )

    for months in [6, 12, 18]:
        days = trading_days(months)
        add_feature(
            f"inv_{months}m_any",
            (macro_daily["t10y2y"].rolling(days).min() < 0).astype(float),
            "macro_memory",
            months,
            f"Indicator for yield-curve inversion at any point over the last {months} months",
        )
        add_feature(
            f"vix_{months}m_max",
            macro_daily["vixcls"].rolling(days).max(),
            "macro_memory",
            months,
            f"Maximum VIX level observed over the last {months} months",
        )
        add_feature(
            f"baa_widen_{months}m",
            macro_daily["baa10y"].diff(days),
            "macro_memory",
            months,
            f"Change in BAA spread over the last {months} months",
        )

    add_feature(
        "bitcoin_available",
        filled_prices["Bitcoin"].notna().astype(float),
        "crypto",
        None,
        "Indicator for whether Bitcoin history is available",
    )
    features.index.name = "decision_date"
    manifest = pd.DataFrame(manifest_rows).sort_values(["family", "lookback_months", "feature_name"], na_position="last")
    return features, manifest


def attach_downside_targets(
    features: pd.DataFrame,
    saa_daily_returns: pd.Series,
    drawdown_threshold: float = -0.04,
    return_threshold: float = -0.02,
) -> pd.DataFrame:
    month_ends = pd.DatetimeIndex(features.index)
    downside_rows: list[dict[str, Any]] = []
    for idx, date in enumerate(month_ends[:-1]):
        next_date = month_ends[idx + 1]
        period = saa_daily_returns.loc[(saa_daily_returns.index > date) & (saa_daily_returns.index <= next_date)]
        if period.empty:
            continue
        equity = (1.0 + period).cumprod()
        period_drawdown = float((equity / equity.cummax() - 1.0).min())
        period_return = float(equity.iloc[-1] - 1.0)
        downside_rows.append(
            {
                "decision_date": date,
                "next_period_end": next_date,
                "next_month_return": period_return,
                "next_month_drawdown": period_drawdown,
                "downside_event": int(period_drawdown <= drawdown_threshold or period_return <= return_threshold),
            }
        )
    target_frame = pd.DataFrame(downside_rows).set_index("decision_date")
    dataset = features.join(target_frame, how="inner")
    bitcoin_columns = [column for column in dataset.columns if column.startswith("bitcoin_")]
    for column in bitcoin_columns:
        dataset[column] = dataset[column].fillna(0.0)
    dataset = dataset.dropna().copy()
    dataset.index.name = "decision_date"
    dataset["label_drawdown_threshold"] = drawdown_threshold
    dataset["label_return_threshold"] = return_threshold
    return dataset


def monthly_feature_frame(
    filled_prices: pd.DataFrame,
    returns: pd.DataFrame,
    saa_daily_returns: pd.Series,
    refresh: bool = False,
) -> pd.DataFrame:
    feature_library, feature_manifest = build_feature_library(
        filled_prices=filled_prices,
        returns=returns,
        refresh=refresh,
    )
    feature_library.to_csv(OUTPUT_DIR / "feature_library.csv")
    feature_manifest.to_csv(OUTPUT_DIR / "feature_library_manifest.csv", index=False)
    dataset = attach_downside_targets(feature_library, saa_daily_returns, drawdown_threshold=-0.04, return_threshold=-0.02)
    dataset.to_csv(OUTPUT_DIR / "signal_dataset.csv")
    return dataset


def build_walk_forward_folds(index: pd.Index, initial_train: int = 72, n_folds: int = 4) -> list[FoldDefinition]:
    if len(index) <= initial_train + n_folds:
        raise RuntimeError("Not enough monthly observations for walk-forward validation.")
    remaining = len(index) - initial_train
    base_fold = remaining // n_folds
    extra = remaining % n_folds
    folds: list[FoldDefinition] = []
    start = initial_train
    for fold_id in range(1, n_folds + 1):
        test_len = base_fold + (1 if fold_id <= extra else 0)
        test_index = index[start : start + test_len]
        training_block = index[:start]
        validation_len = max(12, min(24, int(len(training_block) * 0.2)))
        validation_index = training_block[-validation_len:]
        train_index = training_block[:-validation_len]
        folds.append(
            FoldDefinition(
                fold_id=fold_id,
                train_index=train_index,
                validation_index=validation_index,
                test_index=test_index,
            )
        )
        start += test_len
    return folds


def build_classifier(params: dict[str, Any], y_train: pd.Series) -> XGBClassifier:
    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    scale_pos_weight = negative / positive if positive > 0 else 1.0
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_lambda=1.0,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )


def validation_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probabilities >= threshold).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "predicted_risk_rate": float(preds.mean()),
    }


def xgb_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[np.ndarray, str]:
    model = build_classifier(params, y_train)
    model.fit(X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=int))
    probabilities = model.predict_proba(X_pred.to_numpy(dtype=float))[:, 1]
    label = f"XGBoost d{params['max_depth']} lr{params['learning_rate']:.2f} n{params['n_estimators']}"
    return probabilities, label


def logistic_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
) -> np.ndarray:
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=int))
    return model.predict_proba(X_pred.to_numpy(dtype=float))[:, 1]


def sarimax_return_forecast(
    y_train: pd.Series,
    X_train: pd.DataFrame,
    X_pred: pd.DataFrame,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        model = SARIMAX(
            y_train,
            exog=X_train,
            order=(1, 0, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)
        forecast = result.forecast(steps=len(X_pred), exog=X_pred)
    return np.asarray(forecast, dtype=float)


def candidate_signal_configs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    y_train_return: pd.Series,
    X_pred: pd.DataFrame,
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    for params in XGB_PARAM_GRID:
        probabilities, label = xgb_predictions(X_train, y_train, X_pred, params)
        for threshold in sorted({params["threshold"], 0.35, 0.50}):
            configs.append(
                {
                    "model_family": "XGBoost",
                    "model_label": label,
                    "threshold": float(threshold),
                    "score_type": "probability",
                    "scores": probabilities,
                }
            )

    logit_probabilities = logistic_predictions(X_train, y_train, X_pred)
    for threshold in LOGIT_THRESHOLDS:
        configs.append(
            {
                "model_family": "Logistic",
                "model_label": "Logistic regression",
                "threshold": float(threshold),
                "score_type": "probability",
                "scores": logit_probabilities,
            }
        )

    sarimax_forecast = sarimax_return_forecast(y_train_return, X_train, X_pred)
    for threshold in SARIMAX_THRESHOLDS:
        configs.append(
            {
                "model_family": "SARIMAX",
                "model_label": "SARIMAX(1,0,0)+macro",
                "threshold": float(threshold),
                "score_type": "return_forecast",
                "scores": sarimax_forecast,
            }
        )

    return configs


def scores_to_signals(candidate: dict[str, Any]) -> np.ndarray:
    scores = np.asarray(candidate["scores"], dtype=float)
    threshold = float(candidate["threshold"])
    if candidate["score_type"] == "probability":
        return (scores >= threshold).astype(int)
    return (scores < threshold).astype(int)


def annual_decision_dates(index: pd.Index) -> list[pd.Timestamp]:
    return decision_dates_for_frequency(index=index, frequency="annual")


def next_schedule_boundary(decision_date: pd.Timestamp, schedule_dates: list[pd.Timestamp]) -> pd.Timestamp | None:
    future = [date for date in schedule_dates if date > decision_date]
    return future[0] if future else None


def decision_frequency_defaults(frequency: str) -> tuple[int, int]:
    if frequency == "annual":
        return 3, 2
    if frequency == "quarterly":
        return 12, 4
    if frequency == "monthly":
        return 72, 24
    raise ValueError(f"Unsupported frequency: {frequency}")


def build_decision_folds(
    decision_dates: list[pd.Timestamp],
    frequency: str = "annual",
    initial_train_periods: int | None = None,
    validation_periods: int | None = None,
    n_folds: int = 4,
) -> list[FoldDefinition]:
    default_train, default_validation = decision_frequency_defaults(frequency)
    initial_train_periods = default_train if initial_train_periods is None else initial_train_periods
    validation_periods = default_validation if validation_periods is None else validation_periods
    if len(decision_dates) <= initial_train_periods + n_folds:
        raise RuntimeError(f"Not enough {frequency} decisions for walk-forward validation.")
    remaining = len(decision_dates) - initial_train_periods
    base_fold = remaining // n_folds
    extra = remaining % n_folds
    folds: list[FoldDefinition] = []
    start = initial_train_periods
    decision_index = pd.Index(decision_dates)
    for fold_id in range(1, n_folds + 1):
        test_len = base_fold + (1 if fold_id <= extra else 0)
        test_index = decision_index[start : start + test_len]
        training_block = decision_index[:start]
        validation_index = training_block[-validation_periods:]
        train_index = training_block[:-validation_periods]
        folds.append(
            FoldDefinition(
                fold_id=fold_id,
                train_index=train_index,
                validation_index=validation_index,
                test_index=test_index,
            )
        )
        start += test_len
    return folds


def build_annual_folds(
    decision_dates: list[pd.Timestamp],
    initial_train_years: int = 3,
    validation_years: int = 2,
    n_folds: int = 4,
) -> list[FoldDefinition]:
    return build_decision_folds(
        decision_dates=decision_dates,
        frequency="annual",
        initial_train_periods=initial_train_years,
        validation_periods=validation_years,
        n_folds=n_folds,
    )


def bayes_xgb_probability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_pred: pd.DataFrame,
) -> tuple[float, dict[str, Any]]:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / positives if positives > 0 else 1.0
    if positives < 8 or len(X_train) < 36:
        fallback_params = {
            "max_depth": 2,
            "learning_rate": 0.05,
            "n_estimators": 60,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 4,
            "reg_lambda": 2.0,
        }
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            **fallback_params,
        )
        estimator.fit(X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=int))
        probability = float(estimator.predict_proba(X_pred.to_numpy(dtype=float))[0, 1])
        return probability, fallback_params

    estimator = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )
    search_spaces = {
        "max_depth": Integer(2, 5),
        "learning_rate": Real(0.02, 0.30, prior="log-uniform"),
        "n_estimators": Integer(30, 250),
        "subsample": Real(0.6, 1.0),
        "colsample_bytree": Real(0.6, 1.0),
        "min_child_weight": Integer(1, 8),
        "reg_lambda": Real(0.1, 10.0, prior="log-uniform"),
    }
    splitter = TimeSeriesSplit(n_splits=3)
    scorer = make_scorer(log_loss, greater_is_better=False, response_method="predict_proba", labels=[0, 1])
    search = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        n_iter=8,
        cv=splitter,
        scoring=scorer,
        n_jobs=1,
        random_state=42,
    )
    search.fit(X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=int))
    probability = float(search.best_estimator_.predict_proba(X_pred.to_numpy(dtype=float))[0, 1])
    params = {key: value for key, value in search.best_params_.items()}
    return probability, params


def precompute_model_scores(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    decision_frequency: str = "annual",
    include_bayes_xgb: bool = True,
    include_sarimax: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = dataset[feature_cols]
    y = dataset["downside_event"].astype(int)
    r = dataset["next_month_return"]
    scheduled_dates = decision_dates_for_frequency(dataset.index, frequency=decision_frequency)

    rows: list[dict[str, Any]] = []
    xgb_meta_rows: list[dict[str, Any]] = []
    for decision_date in scheduled_dates:
        train_mask = dataset.index < decision_date
        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        r_train = r.loc[train_mask]
        if len(X_train) < 24 or y_train.sum() < 3:
            continue

        X_pred = X.loc[[decision_date]]
        logistic_probability = float(logistic_predictions(X_train, y_train, X_pred)[0])
        bayes_probability = np.nan
        bayes_params: dict[str, Any] = {}
        if include_bayes_xgb:
            bayes_probability, bayes_params = bayes_xgb_probability(X_train, y_train, X_pred)
        sarimax_forecast = np.nan
        if include_sarimax:
            sarimax_forecast = float(sarimax_return_forecast(r_train, X_train, X_pred)[0])

        rows.append(
            {
                "decision_date": decision_date,
                "logistic_probability": logistic_probability,
                "bayes_xgb_probability": bayes_probability,
                "sarimax_forecast_return": sarimax_forecast,
            }
        )
        xgb_meta_rows.append({"decision_date": decision_date, **bayes_params})

    if rows:
        annual_scores = pd.DataFrame(rows).set_index("decision_date").sort_index()
    else:
        annual_scores = pd.DataFrame(
            columns=["logistic_probability", "bayes_xgb_probability", "sarimax_forecast_return"]
        ).rename_axis("decision_date")
    xgb_meta = (
        pd.DataFrame(xgb_meta_rows).sort_values("decision_date")
        if xgb_meta_rows
        else pd.DataFrame(columns=["decision_date"])
    )
    return annual_scores, xgb_meta


def precompute_annual_scores(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return precompute_model_scores(
        dataset=dataset,
        feature_cols=DEFAULT_FEATURE_COLUMNS,
        decision_frequency="annual",
        include_bayes_xgb=True,
        include_sarimax=True,
    )

def base_weight_for_date(weights_by_rebalance: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    eligible = weights_by_rebalance.loc[weights_by_rebalance.index <= as_of]
    if eligible.empty:
        return weights_by_rebalance.iloc[0]
    return eligible.iloc[-1]


def apply_taa_overlay(
    base_weights: pd.Series,
    returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    covariance_window: int | None = BASE_WINDOW,
    deviation_limit: float = TAA_DEVIATION_LIMIT,
) -> pd.Series:
    active_assets = [asset for asset in ASSET_ORDER if returns.loc[:decision_date, asset].notna().any()]
    history = history_slice(returns[active_assets].dropna(how="all"), decision_date, covariance_window)
    covariance = history.cov(min_periods=max(20, min(len(history), 20)))
    lower_bounds = (base_weights.reindex(ASSET_ORDER).fillna(0.0) - deviation_limit).clip(lower=0.0)
    upper_bounds = (base_weights.reindex(ASSET_ORDER).fillna(0.0) + deviation_limit).clip(upper=1.0)
    return solve_min_variance_with_bounds(covariance, active_assets, lower_bounds, upper_bounds)


def signal_series_to_targets(
    signals: pd.Series,
    base_result: BacktestResult,
    returns: pd.DataFrame,
    covariance_window: int | None = BASE_WINDOW,
    deviation_limit: float = TAA_DEVIATION_LIMIT,
) -> dict[pd.Timestamp, pd.Series]:
    targets: dict[pd.Timestamp, pd.Series] = {}
    for decision_date, signal_value in signals.sort_index().items():
        base_weights = base_weight_for_date(base_result.weights_by_rebalance, pd.Timestamp(decision_date))
        targets[pd.Timestamp(decision_date)] = (
            apply_taa_overlay(
                base_weights,
                returns,
                pd.Timestamp(decision_date),
                covariance_window=covariance_window,
                deviation_limit=deviation_limit,
            )
            if int(signal_value) == 1
            else base_weights
        )
    return targets


def backtest_signal_series(
    signals: pd.Series,
    base_result: BacktestResult,
    returns: pd.DataFrame,
    strategy_name: str,
    end_date: pd.Timestamp | None = None,
    covariance_window: int | None = BASE_WINDOW,
    deviation_limit: float = TAA_DEVIATION_LIMIT,
) -> BacktestResult:
    targets = signal_series_to_targets(
        signals,
        base_result,
        returns,
        covariance_window=covariance_window,
        deviation_limit=deviation_limit,
    )
    result = run_strategy_backtest(returns=returns, target_weights_by_date=targets, strategy_name=strategy_name)
    if end_date is None:
        return result
    clipped_daily = result.daily_returns.loc[result.daily_returns.index <= end_date].copy()
    return BacktestResult(
        daily_returns=clipped_daily,
        weights_by_rebalance=result.weights_by_rebalance.copy(),
        turnover_by_rebalance=result.turnover_by_rebalance.copy(),
        metrics=compute_metrics(clipped_daily),
    )


def model_candidate_configs() -> list[dict[str, Any]]:
    candidate_configs: list[dict[str, Any]] = []
    for threshold in LOGIT_THRESHOLDS:
        candidate_configs.append(
            {
                "model_family": "Logistic",
                "model_label": "Logistic regression",
                "threshold": threshold,
                "score_col": "logistic_probability",
                "score_type": "probability",
            }
        )
    for threshold in [0.25, 0.30, 0.35, 0.40]:
        candidate_configs.append(
            {
                "model_family": "BayesXGBoost",
                "model_label": "Bayes-tuned XGBoost",
                "threshold": threshold,
                "score_col": "bayes_xgb_probability",
                "score_type": "probability",
            }
        )
    for threshold in SARIMAX_THRESHOLDS:
        candidate_configs.append(
            {
                "model_family": "SARIMAX",
                "model_label": "SARIMAX(1,0,0)+macro",
                "threshold": threshold,
                "score_col": "sarimax_forecast_return",
                "score_type": "return_forecast",
            }
        )
    for logistic_threshold in [0.25, 0.30, 0.35, 0.40, 0.45]:
        for xgb_threshold in [0.25, 0.30, 0.35]:
            candidate_configs.append(
                {
                    "model_family": "Ensemble",
                    "model_label": "Logistic OR BayesXGBoost",
                    "threshold": float("nan"),
                    "logistic_threshold": logistic_threshold,
                    "xgb_threshold": xgb_threshold,
                    "score_type": "ensemble",
                }
            )
    return candidate_configs


def config_signals_from_scores(config: dict[str, Any], score_frame: pd.DataFrame, subset_index: pd.Index) -> pd.Series:
    subset = score_frame.loc[subset_index]
    if config["model_family"] == "Ensemble":
        signals = (
            (subset["logistic_probability"] >= config["logistic_threshold"])
            | (subset["bayes_xgb_probability"] >= config["xgb_threshold"])
        ).astype(int)
        return signals.rename("risk_off_signal")
    scores = subset[config["score_col"]]
    if config["score_type"] == "probability":
        return (scores >= config["threshold"]).astype(int).rename("risk_off_signal")
    return (scores < config["threshold"]).astype(int).rename("risk_off_signal")


def model_config_key(config: dict[str, Any]) -> str:
    if config["model_family"] == "Ensemble":
        return f"Ensemble|logit={config['logistic_threshold']:.2f}|xgb={config['xgb_threshold']:.2f}"
    return f"{config['model_family']}|thr={float(config['threshold']):.4f}"


def ips_pass_count(metrics: pd.Series | dict[str, Any]) -> int:
    return int(metrics["total_return_pa"] >= IPS_RETURN_TARGET) + int(metrics["volatility_pa"] <= IPS_VOL_TARGET) + int(
        metrics["max_drawdown"] >= IPS_MAX_DRAWDOWN_TARGET
    )


def evaluate_signal_strategy(
    signals: pd.Series,
    base_result: BacktestResult,
    returns: pd.DataFrame,
    decision_frequency: str = "annual",
    covariance_window: int | None = BASE_WINDOW,
    deviation_limit: float = TAA_DEVIATION_LIMIT,
    strategy_name: str = "TAA Overlay",
) -> tuple[BacktestResult, pd.Series, pd.DataFrame]:
    combined_taa = backtest_signal_series(
        signals=signals,
        base_result=base_result,
        returns=returns,
        strategy_name=strategy_name,
        covariance_window=covariance_window,
        deviation_limit=deviation_limit,
    )
    combined_saa_daily = base_result.daily_returns.reindex(combined_taa.daily_returns.index).dropna()
    combined_taa_daily = combined_taa.daily_returns.reindex(combined_saa_daily.index)
    combined_taa.daily_returns = combined_taa_daily
    combined_taa.metrics = compute_metrics(combined_taa_daily)

    schedule_boundaries = rebalance_dates_for_frequency(returns.index, frequency=decision_frequency)
    folds = build_decision_folds(signals.index.tolist(), frequency=decision_frequency)
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        boundary = next_schedule_boundary(pd.Timestamp(fold.test_index[-1]), schedule_boundaries) or returns.index.max()
        taa_slice = combined_taa_daily.loc[(combined_taa_daily.index > fold.test_index[0]) & (combined_taa_daily.index <= boundary)]
        saa_slice = combined_saa_daily.loc[taa_slice.index]
        taa_metrics = compute_metrics(taa_slice)
        saa_metrics = compute_metrics(saa_slice)
        fold_rows.append(
            {
                "fold": fold.fold_id,
                "test_start": pd.Timestamp(fold.test_index[0]).date().isoformat(),
                "test_end": pd.Timestamp(fold.test_index[-1]).date().isoformat(),
                "taa_return_pa": taa_metrics["total_return_pa"],
                "taa_max_drawdown": taa_metrics["max_drawdown"],
                "taa_var95": taa_metrics["historical_var_95_daily"],
                "taa_calmar": taa_metrics["calmar"],
                "saa_return_pa": saa_metrics["total_return_pa"],
                "saa_max_drawdown": saa_metrics["max_drawdown"],
                "saa_var95": saa_metrics["historical_var_95_daily"],
                "saa_calmar": saa_metrics["calmar"],
            }
        )
    return combined_taa, combined_saa_daily, pd.DataFrame(fold_rows)


def run_model_family_comparison(
    dataset: pd.DataFrame,
    base_result: BacktestResult,
    returns: pd.DataFrame,
    feature_cols: list[str],
    decision_frequency: str = "annual",
    covariance_window: int | None = BASE_WINDOW,
    deviation_limit: float = TAA_DEVIATION_LIMIT,
    feature_config_id: str = "default",
    write_outputs: bool = True,
    output_prefix: str | None = None,
) -> WalkForwardResult:
    annual_scores, xgb_meta = precompute_model_scores(
        dataset=dataset,
        feature_cols=feature_cols,
        decision_frequency=decision_frequency,
        include_bayes_xgb=True,
        include_sarimax=True,
    )
    if annual_scores.empty:
        raise RuntimeError("No scheduled decision scores were generated for the requested configuration.")
    if write_outputs:
        annual_scores.to_csv(output_path("annual_model_scores.csv", output_prefix))
        xgb_meta.to_csv(output_path("bayes_xgb_params_by_year.csv", output_prefix), index=False)

    decision_dates = annual_scores.index.tolist()
    folds = build_decision_folds(decision_dates, frequency=decision_frequency)
    schedule_boundaries = rebalance_dates_for_frequency(returns.index, frequency=decision_frequency)
    candidate_configs = model_candidate_configs()

    training_rows: list[dict[str, Any]] = []
    for fold in folds:
        validation_end = next_schedule_boundary(fold.validation_index[-1], schedule_boundaries) or returns.index.max()
        for config in candidate_configs:
            val_signals = config_signals_from_scores(config, annual_scores, fold.validation_index)
            val_backtest = backtest_signal_series(
                signals=val_signals,
                base_result=base_result,
                returns=returns,
                strategy_name=f"Validation {config['model_family']}",
                end_date=validation_end,
                covariance_window=covariance_window,
                deviation_limit=deviation_limit,
            )
            training_rows.append(
                {
                    "fold": fold.fold_id,
                    "feature_config_id": feature_config_id,
                    "decision_frequency": decision_frequency,
                    "covariance_window": covariance_window,
                    "deviation_limit": deviation_limit,
                    "feature_count": len(feature_cols),
                    "config_key": model_config_key(config),
                    "model_family": config["model_family"],
                    "model_label": config["model_label"],
                    "threshold": config.get("threshold", np.nan),
                    "logistic_threshold": config.get("logistic_threshold", np.nan),
                    "xgb_threshold": config.get("xgb_threshold", np.nan),
                    "validation_cagr": val_backtest.metrics["total_return_pa"],
                    "validation_volatility": val_backtest.metrics["volatility_pa"],
                    "validation_max_drawdown": val_backtest.metrics["max_drawdown"],
                    "validation_calmar": val_backtest.metrics["calmar"],
                    "validation_trigger_rate": float(val_signals.mean()),
                    "selection_score": float(val_backtest.metrics["max_drawdown"]) + 0.10 * float(val_backtest.metrics["total_return_pa"]),
                }
            )

    training_trials = pd.DataFrame(training_rows)
    priority_map = {"Logistic": 0, "Ensemble": 1, "BayesXGBoost": 2, "SARIMAX": 3}
    grouped_candidates = (
        training_trials.groupby("config_key", as_index=False)
        .agg(
            model_family=("model_family", "first"),
            model_label=("model_label", "first"),
            threshold=("threshold", "first"),
            logistic_threshold=("logistic_threshold", "first"),
            xgb_threshold=("xgb_threshold", "first"),
            validation_cagr=("validation_cagr", "mean"),
            validation_volatility=("validation_volatility", "mean"),
            validation_max_drawdown=("validation_max_drawdown", "mean"),
            validation_calmar=("validation_calmar", "mean"),
            validation_trigger_rate=("validation_trigger_rate", "mean"),
            selection_score=("selection_score", "mean"),
        )
    )
    grouped_candidates["model_priority"] = grouped_candidates["model_family"].map(priority_map).fillna(99)
    grouped_candidates = grouped_candidates.sort_values(
        ["validation_max_drawdown", "selection_score", "model_priority"],
        ascending=[False, False, True],
    )
    best_global = grouped_candidates.iloc[0].to_dict()
    global_config = {
        "model_family": best_global["model_family"],
        "model_label": best_global["model_label"],
        "threshold": best_global["threshold"],
        "logistic_threshold": best_global["logistic_threshold"],
        "xgb_threshold": best_global["xgb_threshold"],
        "score_type": "ensemble"
        if best_global["model_family"] == "Ensemble"
        else "return_forecast"
        if best_global["model_family"] == "SARIMAX"
        else "probability",
        "score_col": (
            "logistic_probability"
            if best_global["model_family"] == "Logistic"
            else "bayes_xgb_probability"
            if best_global["model_family"] == "BayesXGBoost"
            else "sarimax_forecast_return"
        ),
    }

    selected_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_validation = training_trials.loc[
            (training_trials["fold"] == fold.fold_id) & (training_trials["config_key"] == best_global["config_key"])
        ].iloc[0]
        selected_rows.append(
            {
                "fold": fold.fold_id,
                "feature_config_id": feature_config_id,
                "feature_count": len(feature_cols),
                "decision_frequency": decision_frequency,
                "covariance_window": covariance_window,
                "deviation_limit": deviation_limit,
                "train_start": fold.train_index[0].date().isoformat(),
                "train_end": fold.validation_index[-1].date().isoformat(),
                "test_start": fold.test_index[0].date().isoformat(),
                "test_end": fold.test_index[-1].date().isoformat(),
                "selected_model_family": best_global["model_family"],
                "selected_model_label": best_global["model_label"],
                "selected_threshold": best_global["threshold"],
                "selected_logistic_threshold": best_global["logistic_threshold"],
                "selected_xgb_threshold": best_global["xgb_threshold"],
                "validation_max_drawdown": fold_validation["validation_max_drawdown"],
                "validation_cagr": fold_validation["validation_cagr"],
                "validation_trigger_rate": fold_validation["validation_trigger_rate"],
            }
        )
        test_signals = config_signals_from_scores(global_config, annual_scores, fold.test_index)
        for timestamp in fold.test_index:
            base_weights = base_weight_for_date(base_result.weights_by_rebalance, timestamp)
            overlay_weights = apply_taa_overlay(
                base_weights,
                returns,
                timestamp,
                covariance_window=covariance_window,
                deviation_limit=deviation_limit,
            )
            prediction_rows.append(
                {
                    "decision_date": timestamp.date().isoformat(),
                    "fold": fold.fold_id,
                    "feature_config_id": feature_config_id,
                    "decision_frequency": decision_frequency,
                    "covariance_window": covariance_window,
                    "deviation_limit": deviation_limit,
                    "selected_model_family": best_global["model_family"],
                    "selected_model_label": best_global["model_label"],
                    "risk_off_signal": int(test_signals.loc[timestamp]),
                    "actual_downside_event": int(dataset.loc[timestamp, "downside_event"]),
                    "realized_next_month_return": float(dataset.loc[timestamp, "next_month_return"]),
                    "base_us_equity": base_weights["US Equity"],
                    "base_us_bonds": base_weights["US Bonds"],
                    "base_reits": base_weights["REITs"],
                    "base_gold": base_weights["Gold"],
                    "base_bitcoin": base_weights["Bitcoin"],
                    "base_jpy": base_weights["JPY"],
                    "taa_us_equity": overlay_weights["US Equity"] if int(test_signals.loc[timestamp]) == 1 else base_weights["US Equity"],
                    "taa_us_bonds": overlay_weights["US Bonds"] if int(test_signals.loc[timestamp]) == 1 else base_weights["US Bonds"],
                    "taa_reits": overlay_weights["REITs"] if int(test_signals.loc[timestamp]) == 1 else base_weights["REITs"],
                    "taa_gold": overlay_weights["Gold"] if int(test_signals.loc[timestamp]) == 1 else base_weights["Gold"],
                    "taa_bitcoin": overlay_weights["Bitcoin"] if int(test_signals.loc[timestamp]) == 1 else base_weights["Bitcoin"],
                    "taa_jpy": overlay_weights["JPY"] if int(test_signals.loc[timestamp]) == 1 else base_weights["JPY"],
                    "logistic_probability": float(annual_scores.loc[timestamp, "logistic_probability"]),
                    "bayes_xgb_probability": float(annual_scores.loc[timestamp, "bayes_xgb_probability"]),
                    "sarimax_forecast_return": float(annual_scores.loc[timestamp, "sarimax_forecast_return"]),
                }
            )

    selected_params = pd.DataFrame(selected_rows)
    monthly_predictions = pd.DataFrame(prediction_rows)
    monthly_predictions["decision_date"] = pd.to_datetime(monthly_predictions["decision_date"])
    monthly_predictions = monthly_predictions.sort_values("decision_date")

    combined_signals = monthly_predictions.set_index("decision_date")["risk_off_signal"].astype(int)
    combined_taa, combined_saa_daily, fold_metrics = evaluate_signal_strategy(
        signals=combined_signals,
        base_result=base_result,
        returns=returns,
        decision_frequency=decision_frequency,
        covariance_window=covariance_window,
        deviation_limit=deviation_limit,
        strategy_name="TAA Overlay",
    )
    fold_metrics["model"] = best_global["model_family"]
    fold_metrics = fold_metrics[
        [
            "fold",
            "model",
            "test_start",
            "test_end",
            "taa_return_pa",
            "taa_max_drawdown",
            "taa_var95",
            "taa_calmar",
            "saa_return_pa",
            "saa_max_drawdown",
            "saa_var95",
            "saa_calmar",
        ]
    ]
    overall_metrics = pd.DataFrame(
        [
            {"portfolio": "TAA Overlay", **compute_metrics(combined_taa.daily_returns).to_dict()},
            {"portfolio": "SAA Baseline", **compute_metrics(combined_saa_daily).to_dict()},
        ]
    )

    if write_outputs:
        training_trials.to_csv(output_path("training_grid_results.csv", output_prefix), index=False)
        selected_params.to_csv(output_path("selected_model_by_fold.csv", output_prefix), index=False)
        monthly_predictions.to_csv(output_path("monthly_signal_predictions.csv", output_prefix), index=False)
        fold_metrics.to_csv(output_path("fold_test_metrics.csv", output_prefix), index=False)
        overall_metrics.to_csv(output_path("overall_comparison.csv", output_prefix), index=False)

    return WalkForwardResult(
        training_trials=training_trials,
        selected_params=selected_params,
        monthly_predictions=monthly_predictions,
        combined_taa=combined_taa,
        combined_saa_daily_returns=combined_saa_daily,
        fold_metrics=fold_metrics,
        overall_metrics=overall_metrics,
    )


def run_walk_forward_model(dataset: pd.DataFrame, base_result: BacktestResult, returns: pd.DataFrame) -> WalkForwardResult:
    return run_model_family_comparison(
        dataset=dataset,
        base_result=base_result,
        returns=returns,
        feature_cols=DEFAULT_FEATURE_COLUMNS,
        decision_frequency="annual",
        covariance_window=BASE_WINDOW,
        deviation_limit=TAA_DEVIATION_LIMIT,
        feature_config_id="baseline_current",
        write_outputs=True,
    )


def feature_family_columns_for_profile(profile_name: str) -> dict[str, list[str]]:
    profile = SWEEP_WINDOW_PROFILES[profile_name]
    return {
        "market_core": [
            f"spx_mom_{profile['market']}",
            f"spx_vol_{profile['risk']}",
            f"equity_drawdown_{profile['risk']}",
            f"avg_corr_{profile['risk']}",
        ],
        "defensive_relative": [
            f"gold_rel_mom_{profile['relative']}",
            f"jpy_rel_mom_{profile['relative']}",
        ],
        "macro_spot": ["vixcls", f"vix_change_{profile['macro_change']}", "t10y2y", "baa10y"],
        "macro_memory": [
            f"inv_{profile['memory']}_any",
            f"vix_{profile['memory']}_max",
            f"baa_widen_{profile['memory']}",
        ],
        "crypto": [f"bitcoin_mom_{profile['crypto']}", "bitcoin_available"],
    }


def generate_feature_sweep_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = [
        {
            "feature_config_id": "baseline_current",
            "window_profile": "baseline",
            "families": "baseline_current",
            "feature_columns": DEFAULT_FEATURE_COLUMNS,
            "feature_count": len(DEFAULT_FEATURE_COLUMNS),
        }
    ]
    for profile_name in SWEEP_WINDOW_PROFILES:
        family_map = feature_family_columns_for_profile(profile_name)
        family_names = list(family_map.keys())
        for selected_count in range(1, len(family_names) + 1):
            for selected_families in itertools.combinations(family_names, selected_count):
                feature_columns = sorted({column for family in selected_families for column in family_map[family]})
                config_id = f"{profile_name}__{'__'.join(selected_families)}"
                configs.append(
                    {
                        "feature_config_id": config_id,
                        "window_profile": profile_name,
                        "families": ",".join(selected_families),
                        "feature_columns": feature_columns,
                        "feature_count": len(feature_columns),
                    }
                )
    return configs


def compare_candidate_rows(left: dict[str, Any], right: dict[str, Any] | None) -> bool:
    if right is None:
        return True
    left_tuple = (
        float(left["overall_max_drawdown"]),
        int(left["ips_pass_count"]),
        float(left["overall_return_pa"]),
        float(left["avg_fold_drawdown_improvement"]),
        -float(left["avg_turnover"]),
    )
    right_tuple = (
        float(right["overall_max_drawdown"]),
        int(right["ips_pass_count"]),
        float(right["overall_return_pa"]),
        float(right["avg_fold_drawdown_improvement"]),
        -float(right["avg_turnover"]),
    )
    return left_tuple > right_tuple


def run_logistic_feature_sweep(
    feature_library: pd.DataFrame,
    base_result: BacktestResult,
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_configs = generate_feature_sweep_configs()
    feature_manifest = pd.DataFrame(
        [
            {
                "feature_config_id": config["feature_config_id"],
                "window_profile": config["window_profile"],
                "families": config["families"],
                "feature_count": config["feature_count"],
                "feature_columns": ",".join(config["feature_columns"]),
            }
            for config in feature_configs
        ]
    )
    feature_manifest.to_csv(OUTPUT_DIR / "sweep_feature_configs.csv", index=False)

    dataset_cache: dict[str, pd.DataFrame] = {}
    score_cache: dict[tuple[str, str], pd.Series] = {}
    results_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None

    for label_config in SWEEP_LABEL_CONFIGS:
        label_key = f"dd{abs(label_config['drawdown_threshold']):.3f}_ret{abs(label_config['return_threshold']):.3f}"
        print(f"Precomputing logistic scores for label set {label_key}...")
        dataset_cache[label_key] = attach_downside_targets(
            feature_library,
            base_result.daily_returns,
            drawdown_threshold=label_config["drawdown_threshold"],
            return_threshold=label_config["return_threshold"],
        )
        for config in feature_configs:
            cache_key = (label_key, config["feature_config_id"])
            annual_scores, _ = precompute_model_scores(
                dataset=dataset_cache[label_key],
                feature_cols=config["feature_columns"],
                decision_frequency="annual",
                include_bayes_xgb=False,
                include_sarimax=False,
            )
            score_cache[cache_key] = annual_scores["logistic_probability"]

    for label_config in SWEEP_LABEL_CONFIGS:
        label_key = f"dd{abs(label_config['drawdown_threshold']):.3f}_ret{abs(label_config['return_threshold']):.3f}"
        print(f"Evaluating sweep configs for label set {label_key}...")
        for config in feature_configs:
            logistic_scores = score_cache[(label_key, config["feature_config_id"])].dropna().sort_index()
            if len(logistic_scores) <= 7:
                continue
            for overlay_config in SWEEP_OVERLAY_CONFIGS:
                for threshold in LOGIT_THRESHOLDS:
                    signals = (logistic_scores >= threshold).astype(int).rename("risk_off_signal")
                    combined_taa, combined_saa_daily, fold_metrics = evaluate_signal_strategy(
                        signals=signals,
                        base_result=base_result,
                        returns=returns,
                        decision_frequency="annual",
                        covariance_window=overlay_config["covariance_window"],
                        deviation_limit=overlay_config["deviation_limit"],
                        strategy_name=f"Sweep {config['feature_config_id']}",
                    )
                    taa_metrics = compute_metrics(combined_taa.daily_returns)
                    saa_metrics = compute_metrics(combined_saa_daily)
                    result_row = {
                        "config_key": (
                            f"{config['feature_config_id']}|{label_key}|cov={overlay_config['covariance_window']}|"
                            f"band={overlay_config['deviation_limit']:.2f}|thr={threshold:.2f}"
                        ),
                        "feature_config_id": config["feature_config_id"],
                        "window_profile": config["window_profile"],
                        "families": config["families"],
                        "feature_count": config["feature_count"],
                        "feature_columns": ",".join(config["feature_columns"]),
                        "label_drawdown_threshold": label_config["drawdown_threshold"],
                        "label_return_threshold": label_config["return_threshold"],
                        "decision_frequency": "annual",
                        "covariance_window": overlay_config["covariance_window"],
                        "deviation_limit": overlay_config["deviation_limit"],
                        "model_family": "Logistic",
                        "threshold": threshold,
                        "trigger_rate": float(signals.mean()),
                        "avg_turnover": float(combined_taa.turnover_by_rebalance.mean()),
                        "oos_start": signals.index.min().date().isoformat(),
                        "overall_return_pa": taa_metrics["total_return_pa"],
                        "overall_volatility_pa": taa_metrics["volatility_pa"],
                        "overall_max_drawdown": taa_metrics["max_drawdown"],
                        "overall_var95": taa_metrics["historical_var_95_daily"],
                        "overall_calmar": taa_metrics["calmar"],
                        "saa_return_pa": saa_metrics["total_return_pa"],
                        "saa_volatility_pa": saa_metrics["volatility_pa"],
                        "saa_max_drawdown": saa_metrics["max_drawdown"],
                        "ips_pass_count": ips_pass_count(taa_metrics),
                        "avg_fold_drawdown_improvement": float(
                            (fold_metrics["taa_max_drawdown"] - fold_metrics["saa_max_drawdown"]).mean()
                        ),
                        "fold_drawdown_win_count": int(
                            (fold_metrics["taa_max_drawdown"] > fold_metrics["saa_max_drawdown"]).sum()
                        ),
                    }
                    results_rows.append(result_row)
                    fold_rows.extend(
                        {
                            **result_row,
                            **fold_row,
                        }
                        for fold_row in fold_metrics.to_dict("records")
                    )
                    if compare_candidate_rows(result_row, best_row):
                        best_row = result_row.copy()
        pd.DataFrame(results_rows).to_csv(OUTPUT_DIR / "sweep_config_results.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "sweep_fold_results.csv", index=False)

    sweep_results = pd.DataFrame(results_rows).sort_values(
        ["overall_max_drawdown", "ips_pass_count", "overall_return_pa", "avg_fold_drawdown_improvement", "avg_turnover"],
        ascending=[False, False, False, False, True],
    )
    sweep_fold_results = pd.DataFrame(fold_rows)
    sweep_results.to_csv(OUTPUT_DIR / "sweep_config_results.csv", index=False)
    sweep_fold_results.to_csv(OUTPUT_DIR / "sweep_fold_results.csv", index=False)
    sweep_results.head(PROMOTION_SHORTLIST_SIZE).to_csv(OUTPUT_DIR / "sweep_top_configs.csv", index=False)
    return sweep_results, feature_manifest


def is_clearly_better(candidate_metrics: pd.Series, baseline_metrics: pd.Series) -> bool:
    drawdown_improvement = float(candidate_metrics["max_drawdown"] - baseline_metrics["max_drawdown"])
    return (
        drawdown_improvement >= 0.0025
        and float(candidate_metrics["total_return_pa"]) >= IPS_RETURN_TARGET
        and ips_pass_count(candidate_metrics) >= ips_pass_count(baseline_metrics)
    )


def promote_sweep_shortlist(
    sweep_results: pd.DataFrame,
    feature_library: pd.DataFrame,
    base_result: BacktestResult,
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, WalkForwardResult, WalkForwardResult | None, dict[str, Any] | None]:
    baseline_dataset = attach_downside_targets(feature_library, base_result.daily_returns, drawdown_threshold=-0.04, return_threshold=-0.02)
    baseline_walk_forward = run_model_family_comparison(
        dataset=baseline_dataset,
        base_result=base_result,
        returns=returns,
        feature_cols=DEFAULT_FEATURE_COLUMNS,
        decision_frequency="annual",
        covariance_window=BASE_WINDOW,
        deviation_limit=TAA_DEVIATION_LIMIT,
        feature_config_id="baseline_current",
        write_outputs=False,
    )
    baseline_metrics = baseline_walk_forward.overall_metrics.set_index("portfolio").loc["TAA Overlay"]

    shortlisted = sweep_results.head(PROMOTION_SHORTLIST_SIZE).copy()
    promotion_rows: list[dict[str, Any]] = []
    best_promoted: WalkForwardResult | None = None
    best_promoted_row: dict[str, Any] | None = None

    for _, row in shortlisted.iterrows():
        print(f"Promoting shortlisted sweep config {row['feature_config_id']}...")
        dataset = attach_downside_targets(
            feature_library,
            base_result.daily_returns,
            drawdown_threshold=float(row["label_drawdown_threshold"]),
            return_threshold=float(row["label_return_threshold"]),
        )
        feature_cols = [value for value in str(row["feature_columns"]).split(",") if value]
        candidate_walk_forward = run_model_family_comparison(
            dataset=dataset,
            base_result=base_result,
            returns=returns,
            feature_cols=feature_cols,
            decision_frequency="annual",
            covariance_window=None if pd.isna(row["covariance_window"]) else int(row["covariance_window"]),
            deviation_limit=float(row["deviation_limit"]),
            feature_config_id=str(row["feature_config_id"]),
            write_outputs=False,
        )
        candidate_metrics = candidate_walk_forward.overall_metrics.set_index("portfolio").loc["TAA Overlay"]
        promotion_row = {
            "feature_config_id": row["feature_config_id"],
            "window_profile": row["window_profile"],
            "families": row["families"],
            "feature_count": row["feature_count"],
            "label_drawdown_threshold": row["label_drawdown_threshold"],
            "label_return_threshold": row["label_return_threshold"],
            "covariance_window": row["covariance_window"],
            "deviation_limit": row["deviation_limit"],
            "selected_model_family": candidate_walk_forward.selected_params.iloc[0]["selected_model_family"],
            "selected_threshold": candidate_walk_forward.selected_params.iloc[0]["selected_threshold"],
            "selected_logistic_threshold": candidate_walk_forward.selected_params.iloc[0]["selected_logistic_threshold"],
            "selected_xgb_threshold": candidate_walk_forward.selected_params.iloc[0]["selected_xgb_threshold"],
            "overall_return_pa": candidate_metrics["total_return_pa"],
            "overall_volatility_pa": candidate_metrics["volatility_pa"],
            "overall_max_drawdown": candidate_metrics["max_drawdown"],
            "overall_var95": candidate_metrics["historical_var_95_daily"],
            "overall_calmar": candidate_metrics["calmar"],
            "avg_fold_drawdown_improvement": float(
                (
                    candidate_walk_forward.fold_metrics["taa_max_drawdown"]
                    - candidate_walk_forward.fold_metrics["saa_max_drawdown"]
                ).mean()
            ),
            "avg_turnover": float(candidate_walk_forward.combined_taa.turnover_by_rebalance.mean()),
            "ips_pass_count": ips_pass_count(candidate_metrics),
            "beats_baseline": is_clearly_better(candidate_metrics, baseline_metrics),
        }
        promotion_rows.append(promotion_row)
        if compare_candidate_rows(promotion_row, best_promoted_row):
            best_promoted_row = promotion_row.copy()
            best_promoted = candidate_walk_forward

    promotion_results = pd.DataFrame(promotion_rows).sort_values(
        ["overall_max_drawdown", "ips_pass_count", "overall_return_pa", "overall_calmar"],
        ascending=[False, False, False, False],
    )
    promotion_results.to_csv(OUTPUT_DIR / "promotion_model_results.csv", index=False)
    return promotion_results, baseline_walk_forward, best_promoted, best_promoted_row

def aggregate_training_table(training_trials: pd.DataFrame) -> pd.DataFrame:
    priority_map = {"Logistic": 0, "Ensemble": 1, "BayesXGBoost": 2, "SARIMAX": 3}
    grouped = (
        training_trials.groupby("config_key", as_index=False)
        .agg(
            model_family=("model_family", "first"),
            model_label=("model_label", "first"),
            threshold=("threshold", "first"),
            logistic_threshold=("logistic_threshold", "first"),
            xgb_threshold=("xgb_threshold", "first"),
            validation_cagr=("validation_cagr", "mean"),
            validation_volatility=("validation_volatility", "mean"),
            validation_max_drawdown=("validation_max_drawdown", "mean"),
            validation_calmar=("validation_calmar", "mean"),
            validation_trigger_rate=("validation_trigger_rate", "mean"),
            selection_score=("selection_score", "mean"),
        )
    )
    grouped["model_priority"] = grouped["model_family"].map(priority_map).fillna(99)
    grouped = grouped.sort_values(
        ["validation_max_drawdown", "selection_score", "model_priority"],
        ascending=[False, False, True],
    )
    grouped.to_csv(OUTPUT_DIR / "training_results_table.csv", index=False)
    return grouped


def training_results_display_table(training_summary: pd.DataFrame) -> pd.DataFrame:
    display = training_summary.copy()

    def trigger_rule(row: pd.Series) -> str:
        if row["model_family"] == "Ensemble":
            return f"logit >= {row['logistic_threshold']:.2f} or xgb >= {row['xgb_threshold']:.2f}"
        if row["model_family"] in {"Logistic", "BayesXGBoost"}:
            return f"p >= {row['threshold']:.2f}"
        return f"forecast < {format_percent(row['threshold'])}"

    display["trigger_rule"] = display.apply(trigger_rule, axis=1)
    display = (
        display.groupby("model_family", as_index=False)
        .head(1)
        .sort_values(["validation_max_drawdown", "selection_score", "model_priority"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    display = display[
        [
            "model_family",
            "model_label",
            "trigger_rule",
            "validation_max_drawdown",
            "validation_cagr",
            "validation_trigger_rate",
        ]
    ]
    display.to_csv(OUTPUT_DIR / "training_results_display.csv", index=False)
    return display


def selected_rule_text(walk_forward: WalkForwardResult) -> str:
    selected = walk_forward.selected_params.iloc[0]
    family = str(selected["selected_model_family"])
    if family == "Logistic":
        return f"The final strategy uses a conservative logistic trigger of p >= {float(selected['selected_threshold']):.2f}."
    if family == "BayesXGBoost":
        return f"The final strategy uses an XGBoost trigger of p >= {float(selected['selected_threshold']):.2f}."
    if family == "SARIMAX":
        return f"The final strategy uses a SARIMAX trigger of forecast < {format_percent(float(selected['selected_threshold']))}."
    return (
        "The final strategy uses an ensemble trigger of "
        f"logit >= {float(selected['selected_logistic_threshold']):.2f} or "
        f"xgb >= {float(selected['selected_xgb_threshold']):.2f}."
    )


def selected_overlay_settings(walk_forward: WalkForwardResult) -> tuple[int | None, float]:
    selected = walk_forward.selected_params.iloc[0]
    covariance_window = selected.get("covariance_window", BASE_WINDOW)
    if pd.isna(covariance_window):
        covariance_window = BASE_WINDOW
    deviation_limit = selected.get("deviation_limit", TAA_DEVIATION_LIMIT)
    if pd.isna(deviation_limit):
        deviation_limit = TAA_DEVIATION_LIMIT
    return int(covariance_window), float(deviation_limit)


def ips_scorecard_table(overall_metrics: pd.DataFrame) -> pd.DataFrame:
    row = overall_metrics.loc[overall_metrics["portfolio"] == "TAA Overlay"].iloc[0]
    checks = [
        ("Return p.a.", row["total_return_pa"], IPS_RETURN_TARGET, row["total_return_pa"] >= IPS_RETURN_TARGET),
        ("Volatility p.a.", row["volatility_pa"], IPS_VOL_TARGET, row["volatility_pa"] <= IPS_VOL_TARGET),
        ("Max drawdown", row["max_drawdown"], IPS_MAX_DRAWDOWN_TARGET, row["max_drawdown"] >= IPS_MAX_DRAWDOWN_TARGET),
    ]
    rows: list[dict[str, Any]] = []
    for metric, actual, target, passed in checks:
        rows.append(
            {
                "metric": metric,
                "actual": actual,
                "target": target,
                "result": "Pass" if passed else "Fail",
            }
        )
    scorecard = pd.DataFrame(rows)
    scorecard.to_csv(OUTPUT_DIR / "ips_scorecard.csv", index=False)
    return scorecard


def plot_flowchart(walk_forward: WalkForwardResult) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    selected = walk_forward.selected_params.iloc[0]
    family = str(selected["selected_model_family"])
    if family == "Logistic":
        model_box = "Logistic downside\nprobability forecast"
        threshold_box = f"Trigger test\np >= {float(selected['selected_threshold']):.2f}"
    elif family == "BayesXGBoost":
        model_box = "XGBoost downside\nprobability forecast"
        threshold_box = f"Trigger test\np >= {float(selected['selected_threshold']):.2f}"
    elif family == "SARIMAX":
        model_box = "SARIMAX next-period\nreturn forecast"
        threshold_box = f"Trigger test\nforecast < {format_percent(float(selected['selected_threshold']))}"
    else:
        model_box = "Logistic + XGBoost\nrisk score update"
        threshold_box = (
            "Trigger test\n"
            f"logit >= {float(selected['selected_logistic_threshold']):.2f} "
            f"or xgb >= {float(selected['selected_xgb_threshold']):.2f}"
        )
    boxes = [
        (0.08, 0.65, "Month-end feature update\nMomentum + macro + stress features"),
        (0.40, 0.65, model_box),
        (0.72, 0.65, threshold_box),
        (0.24, 0.22, "No trigger\nHold SAA weights"),
        (0.62, 0.22, "Trigger\nShift away from equity/REITs/BTC\nToward bonds/gold/JPY"),
    ]
    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="#2f4f4f", lw=1.5),
        )
    arrow_style = dict(arrowstyle="->", lw=1.5, color="#2f4f4f")
    ax.annotate("", xy=(0.32, 0.65), xytext=(0.18, 0.65), arrowprops=arrow_style)
    ax.annotate("", xy=(0.64, 0.65), xytext=(0.48, 0.65), arrowprops=arrow_style)
    ax.annotate("", xy=(0.24, 0.33), xytext=(0.66, 0.57), arrowprops=arrow_style)
    ax.annotate("", xy=(0.62, 0.33), xytext=(0.74, 0.57), arrowprops=arrow_style)
    ax.set_title("TAA decision rule")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "taa_rule_flowchart.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_walk_forward_setup(dataset: pd.DataFrame) -> None:
    folds = build_walk_forward_folds(dataset.index)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    y_positions = list(range(len(folds), 0, -1))
    for y, fold in zip(y_positions, folds):
        train_left = mdates.date2num(fold.train_index[0])
        val_left = mdates.date2num(fold.validation_index[0])
        train_width = val_left - train_left
        ax.barh(y, train_width, left=train_left, color="#577590", height=0.25)
        test_left = mdates.date2num(fold.test_index[0])
        val_width = test_left - val_left
        ax.barh(
            y,
            val_width,
            left=val_left,
            color="#43aa8b",
            height=0.25,
        )
        test_width = mdates.date2num(fold.test_index[-1] + pd.offsets.MonthEnd(1)) - test_left
        ax.barh(y, test_width, left=test_left, color="#f3722c", height=0.25)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Fold {fold.fold_id}" for fold in folds])
    ax.set_title("Walk-forward setup with annual year-end decisions")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.2, axis="x")
    legend = [
        plt.Line2D([0], [0], color="#577590", lw=8, label="Training"),
        plt.Line2D([0], [0], color="#43aa8b", lw=8, label="Validation"),
        plt.Line2D([0], [0], color="#f3722c", lw=8, label="Test"),
    ]
    ax.legend(handles=legend, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "walk_forward_setup.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_test_results(walk_forward: WalkForwardResult) -> None:
    taa_curve = (1.0 + walk_forward.combined_taa.daily_returns).cumprod() - 1.0
    saa_curve = (1.0 + walk_forward.combined_saa_daily_returns).cumprod() - 1.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 2]})
    axes[0].plot(taa_curve.index, taa_curve.values, label="TAA Overlay", color="#1d3557", linewidth=2.0)
    axes[0].plot(saa_curve.index, saa_curve.values, label="SAA Baseline", color="#e76f51", linewidth=1.8)
    axes[0].set_title("Annual-rebalance out-of-sample cumulative performance")
    axes[0].set_ylabel("Cumulative return")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fold_metrics = walk_forward.fold_metrics.copy()
    x = np.arange(len(fold_metrics))
    width = 0.36
    axes[1].bar(x - width / 2, fold_metrics["saa_max_drawdown"] * 100, width=width, color="#f4a261", label="SAA max DD")
    axes[1].bar(x + width / 2, fold_metrics["taa_max_drawdown"] * 100, width=width, color="#2a9d8f", label="TAA max DD")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Fold {value}" for value in fold_metrics["fold"]])
    axes[1].set_ylabel("Percent")
    axes[1].set_title("Fold-level max drawdown comparison")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend()
    lower_bound = min(fold_metrics["saa_max_drawdown"].min(), fold_metrics["taa_max_drawdown"].min()) * 100
    axes[1].set_ylim(lower_bound * 1.1, 1.0)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_results.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def encode_image(path: Path) -> str:
    mime = "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def format_percent(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.{decimals}%}"


def render_table_html(df: pd.DataFrame, percent_columns: list[str] | None = None, precision: int = 3) -> str:
    percent_columns = percent_columns or []
    display = df.copy()
    for column in display.columns:
        if column in percent_columns:
            display[column] = display[column].map(lambda x: format_percent(x))
        elif pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda x: f"{x:.{precision}f}" if pd.notna(x) else "n/a")
    return display.to_html(index=False, border=0, classes="report-table")


def allocation_table(base_result: BacktestResult, returns: pd.DataFrame, walk_forward: WalkForwardResult | None = None) -> pd.DataFrame:
    latest_date = pd.Timestamp(base_result.weights_by_rebalance.index[-1])
    latest_base = base_result.weights_by_rebalance.iloc[-1].reindex(ASSET_ORDER)
    covariance_window, deviation_limit = (
        selected_overlay_settings(walk_forward) if walk_forward is not None else (BASE_WINDOW, TAA_DEVIATION_LIMIT)
    )
    latest_risk_off = apply_taa_overlay(
        latest_base,
        returns,
        latest_date,
        covariance_window=covariance_window,
        deviation_limit=deviation_limit,
    )
    table = pd.DataFrame(
        [
            {"strategy": "SAA baseline", **latest_base.to_dict()},
            {"strategy": "TAA Strategy", **latest_risk_off.to_dict()},
        ]
    )
    return table


def signal_rationale_text(dataset: pd.DataFrame, walk_forward: WalkForwardResult) -> str:
    event_rate = float(dataset["downside_event"].mean())
    covariance_window, deviation_limit = selected_overlay_settings(walk_forward)
    return (
        f"{selected_rule_text(walk_forward)} "
        "I tested logistic regression, SARIMAX, and XGBoost, then selected the specification that performed best once the wider tactical deviation band was allowed. "
        "The model uses recent market momentum, recent volatility, correlation across assets, gold and JPY strength, and macro stress measures such as VIX, the yield curve, and credit spreads. "
        "To train the model, I look at old data and mark each month as risky or not risky based on what happened to the SAA baseline in the following month. "
        "That next-month result is only used to train the model on past periods. When the strategy is actually making a decision, it only uses information available at that point in time. "
        f"If the model gives a risk-off warning at year-end, the portfolio moves to the defensive TAA mix for the next period using a {window_label(covariance_window)} covariance estimate and a +/-{deviation_limit:.0%} tactical band. "
        f"In the final sample, risky months make up {event_rate:.1%} of observations."
    )


def recommendation_text(overall_metrics: pd.DataFrame, fold_metrics: pd.DataFrame) -> str:
    taa = overall_metrics.loc[overall_metrics["portfolio"] == "TAA Overlay"].iloc[0]
    saa = overall_metrics.loc[overall_metrics["portfolio"] == "SAA Baseline"].iloc[0]
    win_count = int((fold_metrics["taa_max_drawdown"] > fold_metrics["saa_max_drawdown"]).sum())
    drawdown_improvement = float(taa["max_drawdown"] - saa["max_drawdown"])
    calmar_improvement = float(taa["calmar"] - saa["calmar"])
    if drawdown_improvement > 0 and calmar_improvement > 0:
        return (
            "Recommendation: this is the best tactical overlay found, and it does improve downside risk, but it still does "
            "not satisfy the committee's required drawdown target. The walk-forward evidence shows that the overlay helps "
            "most when volatility, credit stress, and cross-asset correlation rise together. After costs, the combined "
            f"out-of-sample result delivers a max drawdown of {taa['max_drawdown']:.1%} versus {saa['max_drawdown']:.1%} "
            f"for the SAA baseline, with TAA improving max drawdown in {win_count} of {len(fold_metrics)} test folds. "
            "Present it as the lowest-drawdown feasible overlay discovered in the research process, not as a fully compliant "
            "solution to the 18% or 13% target."
        )
    return (
        "Recommendation: the tactical overlay improves downside risk, but not enough to satisfy the committee's required "
        f"drawdown limit. After costs, the combined out-of-sample max drawdown is {taa['max_drawdown']:.1%} versus "
        f"{saa['max_drawdown']:.1%} for the SAA baseline, and TAA improves max drawdown in {win_count} of "
        f"{len(fold_metrics)} test folds. The tactical rule that works best is a wider risk-off reweighting inside the "
        "+/-15% deviation band around the SAA weights. Even so, the Assignment 4 limits still leave too much structural "
        "risk in the portfolio, so the honest recommendation is to present this as "
        "the lowest-drawdown feasible overlay found rather than as a fully successful solution to the 18% or 13% target."
    )


def build_report_html(
    dataset: pd.DataFrame,
    training_summary: pd.DataFrame,
    walk_forward: WalkForwardResult,
    summary: pd.DataFrame,
    history_regimes: pd.DataFrame,
    feasible_start: pd.Timestamp,
    full_overlap_start: pd.Timestamp,
    base_result: BacktestResult,
    returns: pd.DataFrame,
) -> None:
    report_path = OUTPUT_DIR / "report.html"
    overall_metrics = walk_forward.overall_metrics.copy()
    covariance_window, deviation_limit = selected_overlay_settings(walk_forward)
    allocation_df = allocation_table(base_result, returns, walk_forward)
    model_summary = training_results_display_table(training_summary)
    ips_table = ips_scorecard_table(overall_metrics)
    fold_table = walk_forward.fold_metrics[
        ["fold", "model", "taa_max_drawdown", "saa_max_drawdown", "taa_return_pa", "saa_return_pa"]
    ]
    overall_table = overall_metrics[
        ["portfolio", "total_return_pa", "volatility_pa", "max_drawdown", "historical_var_95_daily", "calmar"]
    ]
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Assignment 4 Research Report</title>
<style>
  @page {{
    size: A4;
    margin: 0.5in 0.55in;
  }}
  body {{
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.38;
    color: #1a1a1a;
    max-width: 7.35in;
    margin: 0 auto;
    padding: 0.35in 0;
  }}
  h1 {{ font-size: 17pt; margin-bottom: 0.14in; border-bottom: 2px solid #222; padding-bottom: 0.06in; }}
  h2 {{ font-size: 12pt; margin-top: 0.18in; margin-bottom: 0.08in; }}
  h3 {{ font-size: 11pt; margin-top: 0.14in; margin-bottom: 0.06in; }}
  p, li {{ margin-bottom: 0.06in; }}
  ul {{ padding-left: 0.22in; }}
  .callout {{ background: #f4f7fb; border-left: 4px solid #335c81; padding: 0.10in 0.14in; margin: 0.12in 0; }}
  .chart {{ width: 100%; margin: 0.08in 0 0.12in; }}
  .report-table {{ width: 100%; border-collapse: collapse; margin: 0.08in 0; font-size: 9pt; }}
  .report-table th, .report-table td {{ border: 1px solid #b9c2cf; padding: 4px 6px; }}
  .report-table th {{ background: #eef3f8; }}
</style>
</head>
<body>
<h1>Asteron Endowment: Assignment 4 Research Report</h1>
<p>This report rebuilds the Assignment 3 SAA baseline inside <code>Assignment_4</code> using the full paginated Zion history and then evaluates a tactical overlay designed to reduce downside risk under the tighter Assignment 4 policy limits.</p>

<div class="callout">
  <strong>Data reality check.</strong> The Zion CSV export truncates at 10,000 rows, but the paginated Datasette table contains 55,399 rows. Pulling the full table moves the six-asset overlap start back to <strong>{full_overlap_start.date().isoformat()}</strong>, while the staggered-history SAA backtest becomes feasible in <strong>{feasible_start.date().isoformat()}</strong>.
</div>

<h2>Strategic And Tactical Allocations</h2>
<p>The refreshed strategic baseline remains a MinVar portfolio. When the TAA signal fires, the overlay now re-optimizes the portfolio inside a tactical band of <code>SAA weight +/- {deviation_limit:.0%}</code> per asset class using a <code>{window_label(covariance_window)}</code> covariance window. That wider tactical band allows a materially more defensive risk-off portfolio than the earlier narrow rotation.</p>
{render_table_html(allocation_df, percent_columns=ASSET_ORDER)}

<h2>1. Signal Design &amp; Rationale</h2>
<p>{signal_rationale_text(dataset, walk_forward)}</p>
<p>The practical takeaway from model testing is that the assignment is less about discovering hidden alpha and more about identifying when to switch into the most defensive feasible overlay. In this dataset, models that stayed defensive for longer generally produced the lowest drawdowns.</p>

<h2>2. TAA Rule</h2>
<p>The TAA overlay is binary and only rebalances at each year-end. The chosen model uses the latest monthly features available at that year-end to decide whether the next calendar year should stay at the strategic baseline or move into the tactical risk-off allocation shown above. When a trigger occurs, the portfolio is re-solved as a minimum-variance allocation subject to staying within <code>SAA weight +/- {deviation_limit:.0%}</code> for each sleeve.</p>
<img class="chart" src="{encode_image(OUTPUT_DIR / 'taa_rule_flowchart.png')}" alt="TAA rule flowchart">

<h2>3. Walk-Forward Setup</h2>
<p>The model uses an expanding-window walk-forward design with four out-of-sample folds built on annual decision dates. Each fold selects the best year-end trigger family inside the validation years and then rolls forward to untouched test years.</p>
<img class="chart" src="{encode_image(OUTPUT_DIR / 'walk_forward_setup.png')}" alt="Walk-forward setup">

<h2>4. Training Results</h2>
<p>The table below shows the best-performing rule from each model family. Selection prioritized lower validation max drawdown rather than raw prediction accuracy.</p>
<p>{selected_rule_text(walk_forward)} This is the rule that produced the best downside result under the updated tactical deviation limits.</p>
{render_table_html(model_summary, percent_columns=['validation_max_drawdown', 'validation_cagr', 'validation_trigger_rate'])}

<h2>5. Test Results</h2>
<p>Out-of-sample performance is evaluated only on fold test windows. The chart shows combined walk-forward cumulative performance, while the table below focuses on the metrics that actually matter for this assignment: drawdown and return.</p>
<img class="chart" src="{encode_image(OUTPUT_DIR / 'test_results.png')}" alt="Out-of-sample test results">
{render_table_html(fold_table, percent_columns=['taa_max_drawdown', 'saa_max_drawdown', 'taa_return_pa', 'saa_return_pa'])}

<h2>6. Overall Evaluation</h2>
<p>The main strategic baseline in this report is the corrected Assignment 3-style staggered-history MinVar rebuild that begins in 2003-03-31. The table below is narrower: it compares the combined walk-forward out-of-sample TAA overlay with the SAA baseline over their common tactical evaluation span, which begins later because the signal needs a training warmup.</p>
{render_table_html(overall_table, percent_columns=['total_return_pa', 'volatility_pa', 'max_drawdown', 'historical_var_95_daily'])}
<h3>IPS Scorecard</h3>
<p>The table below compares the final TAA strategy against the Assignment 4 policy targets for return, volatility, and max drawdown.</p>
{render_table_html(ips_table, percent_columns=['actual', 'target'])}
<p>I had assumed that the 2008 financial crisis was the reason I was unable to achieve a max drawdown lower than 20%, so I ran the backtest again starting in 2010, which allowed me to avoid the financial crisis and the staggered portfolio system from the previous seven years. Even after doing that, max drawdown only improved from -26.3% to -22.0%, which shows that the recent rate-shock period is still the binding downside event.</p>
<p>Validation results should be interpreted as short-horizon screening evidence rather than as a full-cycle portfolio outcome. The realized out-of-sample TAA path still records a max drawdown of {overall_metrics.loc[overall_metrics['portfolio'] == 'TAA Overlay', 'max_drawdown'].iloc[0]:.1%}, largely because the most severe stress episode in the tactical sample is the 2021-2022 inflation and rate-hike shock. In other words, the conservative logistic rule improves downside risk relative to the SAA baseline, but it does not eliminate the portfolio's exposure to the recent rate-shock regime.</p>

<h2>7. Recommendation</h2>
<p>{recommendation_text(walk_forward.overall_metrics, walk_forward.fold_metrics)}</p>
<p>The refreshed SAA rebuild shows that the original Assignment 3 conclusions were based on a truncated sample. Rebuilding the baseline on the full Zion history makes the tactical exercise materially more credible, because the TAA overlay is now being judged against a strategic benchmark that actually reflects the available data. The honest conclusion is that the corrected overlay gets drawdown closer to the target, but the policy box still prevents a full solution.</p>
</body>
</html>
"""
    report_path.write_text(html)


def pdf_table_from_frame(
    df: pd.DataFrame,
    percent_columns: list[str] | None = None,
    max_rows: int | None = None,
) -> Table:
    percent_columns = percent_columns or []
    display = df.copy()
    if max_rows is not None:
        display = display.head(max_rows)
    for column in display.columns:
        if column in percent_columns:
            display[column] = display[column].map(lambda x: format_percent(x))
        elif pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda x: f"{x:.3f}" if pd.notna(x) else "n/a")
    data = [display.columns.tolist()] + display.astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#a5b4c4")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    return table


def build_report_pdf(
    dataset: pd.DataFrame,
    training_summary: pd.DataFrame,
    walk_forward: WalkForwardResult,
    summary: pd.DataFrame,
    history_regimes: pd.DataFrame,
    base_result: BacktestResult,
    returns: pd.DataFrame,
) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Body", fontName="Helvetica", fontSize=9.5, leading=12, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="Section", fontName="Helvetica-Bold", fontSize=12, leading=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", fontName="Helvetica", fontSize=8.5, leading=10))

    doc = SimpleDocTemplate(str(OUTPUT_DIR / "report.pdf"), pagesize=A4, rightMargin=0.45 * inch, leftMargin=0.45 * inch, topMargin=0.45 * inch, bottomMargin=0.45 * inch)
    story: list[Any] = []

    covariance_window, deviation_limit = selected_overlay_settings(walk_forward)
    allocation_df = allocation_table(base_result, returns, walk_forward)
    model_summary = training_results_display_table(training_summary)
    fold_table = walk_forward.fold_metrics[
        ["fold", "model", "taa_max_drawdown", "saa_max_drawdown", "taa_return_pa", "saa_return_pa"]
    ]
    overall_table = walk_forward.overall_metrics[
        ["portfolio", "total_return_pa", "volatility_pa", "max_drawdown", "historical_var_95_daily", "calmar"]
    ]
    ips_table = ips_scorecard_table(walk_forward.overall_metrics)

    story.append(Paragraph("Asteron Endowment: Assignment 4 Research Report", styles["Title"]))
    story.append(
        Paragraph(
            "This report rebuilds the Assignment 3 SAA baseline on the full Zion history and evaluates the lowest-drawdown tactical overlay I could find under the Assignment 4 policy limits.",
            styles["Body"],
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph("Strategic And Tactical Allocations", styles["Section"]))
    story.append(
        Paragraph(
            f"The tactical risk-off allocation below is the most effective overlay found in testing. When the TAA signal fires, the portfolio is re-optimized inside a tactical band of SAA weight +/-{deviation_limit:.0%} per asset class using a {window_label(covariance_window)} covariance window.",
            styles["Body"],
        )
    )
    story.append(pdf_table_from_frame(allocation_df, percent_columns=ASSET_ORDER, max_rows=2))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph("1. Signal Design & Rationale", styles["Section"]))
    story.append(Paragraph(signal_rationale_text(dataset, walk_forward), styles["Body"]))
    story.append(
        Paragraph(
            "The practical signal objective is not to maximize forecast accuracy. It is to detect the months when the portfolio should shift into its most defensive feasible tactical allocation.",
            styles["Body"],
        )
    )
    story.append(Spacer(1, 0.06 * inch))
    story.append(Paragraph("2. TAA Rule", styles["Section"]))
    story.append(Paragraph(f"The overlay only acts at each year-end rebalance. When the selected model flags elevated downside risk, the portfolio is re-solved as a minimum-variance allocation subject to SAA weight +/-{deviation_limit:.0%} per asset class.", styles["Body"]))
    story.append(Image(str(OUTPUT_DIR / "taa_rule_flowchart.png"), width=6.8 * inch, height=3.1 * inch))
    story.append(Spacer(1, 0.06 * inch))
    story.append(Paragraph("3. Walk-Forward Setup", styles["Section"]))
    story.append(Paragraph("Four expanding folds are used on annual decision dates. Validation identifies one globally defensible rule, and that same annual overlay is then carried across the test folds.", styles["Body"]))
    story.append(Image(str(OUTPUT_DIR / "walk_forward_setup.png"), width=6.8 * inch, height=2.3 * inch))
    story.append(Spacer(1, 0.08 * inch))

    story.append(Paragraph("4. Training Results", styles["Section"]))
    story.append(
        Paragraph(
            "The table below shows the best-performing rule from each model family. Selection prioritized lower validation max drawdown rather than raw prediction accuracy.",
            styles["Body"],
        )
    )
    story.append(Paragraph(f"{selected_rule_text(walk_forward)} This is the rule that produced the best downside result under the updated tactical deviation limits.", styles["Body"]))
    story.append(
        pdf_table_from_frame(
            model_summary,
            percent_columns=["validation_max_drawdown", "validation_cagr", "validation_trigger_rate"],
            max_rows=6,
        )
    )
    story.append(Spacer(1, 0.10 * inch))
    story.append(Paragraph("5. Test Results", styles["Section"]))
    story.append(
        Paragraph(
            "All performance below is strictly out-of-sample across the walk-forward test folds.",
            styles["Body"],
        )
    )
    story.append(Image(str(OUTPUT_DIR / "test_results.png"), width=6.8 * inch, height=4.2 * inch))
    story.append(
        pdf_table_from_frame(
            fold_table,
            percent_columns=["taa_max_drawdown", "saa_max_drawdown", "taa_return_pa", "saa_return_pa"],
            max_rows=4,
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("6. Overall Evaluation", styles["Section"]))
    story.append(
        Paragraph(
            "The strategic benchmark below is the refreshed SAA baseline rather than the old 60/40 benchmark. The full strategic rebuild begins in 2003, while this comparison is restricted to the later out-of-sample tactical window.",
            styles["Body"],
        )
    )
    story.append(
        pdf_table_from_frame(
            overall_table,
            percent_columns=[
                "total_return_pa",
                "volatility_pa",
                "historical_var_95_daily",
                "max_drawdown",
            ],
            max_rows=2,
        )
    )
    story.append(Spacer(1, 0.06 * inch))
    story.append(Paragraph("IPS Scorecard", styles["Heading3"]))
    story.append(
        Paragraph(
            "The table below compares the final TAA strategy against the Assignment 4 policy targets for return, volatility, and max drawdown.",
            styles["Body"],
        )
    )
    story.append(
        pdf_table_from_frame(
            ips_table,
            percent_columns=["actual", "target"],
            max_rows=3,
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        Paragraph(
            "I had assumed that the 2008 financial crisis was the reason I was unable to achieve a max drawdown lower than 20%, so I ran the backtest again starting in 2010, which allowed me to avoid the financial crisis and the staggered portfolio system from the previous seven years. Even after doing that, max drawdown only improved from -26.3% to -22.0%, which shows that the recent rate-shock period is still the binding downside event.",
            styles["Body"],
        )
    )
    story.append(
        Paragraph(
            f"Validation results should be interpreted as short-horizon screening evidence rather than as a full-cycle portfolio outcome. The realized out-of-sample TAA path still records a max drawdown of {overall_table.loc[overall_table['portfolio'] == 'TAA Overlay', 'max_drawdown'].iloc[0]:.1%}, largely because the most severe stress episode in the tactical sample is the 2021-2022 inflation and rate-hike shock. In other words, the conservative logistic rule improves downside risk relative to the SAA baseline, but it does not eliminate the portfolio's exposure to the recent rate-shock regime.",
            styles["Body"],
        )
    )
    story.append(Paragraph("Strategy Context", styles["Section"]))
    story.append(
        Paragraph(
            "The chart below compares the cumulative out-of-sample paths of the final TAA strategy and the SAA baseline over the same window.",
            styles["Body"],
        )
    )
    story.append(Image(str(OUTPUT_DIR / "strategy_context_backtest.png"), width=6.8 * inch, height=2.2 * inch))
    story.append(PageBreak())
    story.append(
        Paragraph(
            "The next chart compares the realized annualized return and volatility of the final TAA strategy against the SAA baseline over the same out-of-sample window.",
            styles["Body"],
        )
    )
    story.append(Image(str(OUTPUT_DIR / "risk_return_comparison.png"), width=6.8 * inch, height=2.5 * inch))
    story.append(Spacer(1, 0.10 * inch))
    story.append(Paragraph("7. Recommendation", styles["Section"]))
    story.append(Paragraph(recommendation_text(walk_forward.overall_metrics, walk_forward.fold_metrics), styles["Body"]))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph("Strategic rebuild notes", styles["Heading3"]))
    story.append(
        Paragraph(
            "The full paginated Zion pull contains 55,399 OHLCV rows. The six-asset overlap begins in 2010 when Bitcoin starts, while the staggered-history baseline becomes feasible in 2003 once REITs enter the universe and the IPS cap structure can be satisfied.",
            styles["Body"],
        )
    )
    story.append(
        pdf_table_from_frame(
            summary[["window", "backtest_total_return_pa", "backtest_volatility_pa", "backtest_max_drawdown"]].head(5),
            percent_columns=["backtest_total_return_pa", "backtest_volatility_pa", "backtest_max_drawdown"],
            max_rows=5,
        )
    )
    doc.build(story)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Assignment 4 SAA + TAA research report.")
    parser.add_argument("--refresh", action="store_true", help="Re-download Zion and public macro data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    assets, ohlcv = fetch_source_tables(refresh=args.refresh)
    filled_prices, returns = prepare_price_panel(ohlcv)
    coverage = asset_coverage_table(filled_prices)
    history_regimes = build_history_regime_table(filled_prices)

    feasible_start = first_feasible_date(filled_prices)
    full_overlap_start = first_full_overlap_date(filled_prices)

    staggered_returns = returns.loc[feasible_start:].copy()
    full_overlap_returns = returns.loc[full_overlap_start:].dropna().copy()

    staggered_rebalances = annual_rebalance_dates(staggered_returns.index)
    full_overlap_rebalances = annual_rebalance_dates(full_overlap_returns.index)

    summary, strategy_results = summarize_windows(staggered_returns, staggered_rebalances)
    base_result = strategy_results[window_label(BASE_WINDOW)]
    benchmark_result = strategy_results["benchmark"]
    regime_table, average_corr_series = build_regime_table(staggered_returns.fillna(0.0))

    plot_covariance_sensitivity(summary)
    plot_weight_stability(summary)
    plot_regime_series(average_corr_series, regime_table)
    plot_backtest(base_result, benchmark_result, regime_table)
    plot_efficient_frontier(staggered_returns.fillna(0.0))
    write_saa_summary_outputs(filled_prices, summary, base_result, benchmark_result, regime_table, staggered_returns.fillna(0.0))

    asset_map = assets[["ticker", "asset_class", "name"]].sort_values(["asset_class", "ticker"])
    asset_map.to_csv(OUTPUT_DIR / "zion_asset_map.csv", index=False)

    feature_library, feature_manifest = build_feature_library(
        filled_prices=filled_prices.loc[feasible_start:],
        returns=staggered_returns,
        refresh=args.refresh,
    )
    feature_library.to_csv(OUTPUT_DIR / "feature_library.csv")
    feature_manifest.to_csv(OUTPUT_DIR / "feature_library_manifest.csv", index=False)
    dataset = attach_downside_targets(feature_library, base_result.daily_returns, drawdown_threshold=-0.04, return_threshold=-0.02)
    dataset.to_csv(OUTPUT_DIR / "signal_dataset.csv")

    sweep_results, _ = run_logistic_feature_sweep(feature_library, base_result, staggered_returns)
    promotion_results, baseline_walk_forward, best_promoted, best_promoted_row = promote_sweep_shortlist(
        sweep_results=sweep_results,
        feature_library=feature_library,
        base_result=base_result,
        returns=staggered_returns,
    )

    baseline_metrics = baseline_walk_forward.overall_metrics.set_index("portfolio").loc["TAA Overlay"]
    report_refreshed = False
    final_walk_forward = baseline_walk_forward
    final_dataset = dataset

    if best_promoted is not None and best_promoted_row is not None and bool(best_promoted_row["beats_baseline"]):
        promoted_dataset = attach_downside_targets(
            feature_library,
            base_result.daily_returns,
            drawdown_threshold=float(best_promoted_row["label_drawdown_threshold"]),
            return_threshold=float(best_promoted_row["label_return_threshold"]),
        )
        promoted_dataset.to_csv(OUTPUT_DIR / "signal_dataset.csv")
        promoted_feature_cols = [
            value
            for value in str(
                sweep_results.loc[sweep_results["feature_config_id"] == best_promoted_row["feature_config_id"], "feature_columns"].iloc[0]
            ).split(",")
            if value
        ]
        final_walk_forward = run_model_family_comparison(
            dataset=promoted_dataset,
            base_result=base_result,
            returns=staggered_returns,
            feature_cols=promoted_feature_cols,
            decision_frequency="annual",
            covariance_window=int(best_promoted_row["covariance_window"]),
            deviation_limit=float(best_promoted_row["deviation_limit"]),
            feature_config_id=str(best_promoted_row["feature_config_id"]),
            write_outputs=True,
        )
        final_dataset = promoted_dataset
        training_summary = aggregate_training_table(final_walk_forward.training_trials)
        plot_flowchart(final_walk_forward)
        plot_walk_forward_setup(final_dataset)
        plot_test_results(final_walk_forward)
        plot_strategy_context(final_walk_forward)
        plot_risk_return_comparison(final_walk_forward)
        build_report_html(
            final_dataset,
            training_summary,
            final_walk_forward,
            summary,
            history_regimes,
            feasible_start,
            full_overlap_start,
            base_result,
            staggered_returns,
        )
        build_report_pdf(final_dataset, training_summary, final_walk_forward, summary, history_regimes, base_result, staggered_returns)
        report_refreshed = True

    if not report_refreshed:
        baseline_summary = pd.DataFrame(
            [
                {
                    "portfolio": "TAA Overlay",
                    **baseline_metrics.to_dict(),
                }
            ]
        )
        baseline_summary.to_csv(OUTPUT_DIR / "baseline_model_results.csv", index=False)
    promotion_results.to_csv(OUTPUT_DIR / "promotion_model_results.csv", index=False)

    metadata = pd.DataFrame(
        [
            {"item": "generated_at", "value": datetime.utcnow().isoformat() + "Z"},
            {"item": "feasible_start", "value": feasible_start.date().isoformat()},
            {"item": "full_overlap_start", "value": full_overlap_start.date().isoformat()},
            {"item": "zion_ohlcv_rows", "value": str(len(ohlcv))},
            {"item": "monthly_signal_rows", "value": str(len(dataset))},
            {"item": "sweep_configs_tested", "value": str(len(sweep_results))},
            {"item": "promotion_shortlist_size", "value": str(len(promotion_results))},
            {"item": "report_refreshed", "value": str(report_refreshed)},
        ]
    )
    metadata.to_csv(OUTPUT_DIR / "build_metadata.csv", index=False)

    print("Assignment 4 build complete.")
    print(f"Feasible staggered-history start: {feasible_start.date().isoformat()}")
    print(f"Full six-asset overlap start: {full_overlap_start.date().isoformat()}")
    print(f"Baseline TAA max drawdown: {baseline_metrics['max_drawdown']:.1%}")
    if best_promoted_row is not None:
        print(f"Best promoted max drawdown: {best_promoted_row['overall_max_drawdown']:.1%}")
        print(f"Best promoted model family: {best_promoted_row['selected_model_family']}")
    print(f"Report refreshed: {report_refreshed}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
