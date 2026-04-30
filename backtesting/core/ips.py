"""Whitmore IPS constants and acceptance helpers.

The IPS PDF is the source of truth. This module encodes the binding
constraints used by the `backtesting/` package:

* 8.0% annualized return objective over rolling five-year periods.
* 15.0% annualized volatility ceiling.
* 25.0% maximum tolerated peak-to-trough drawdown.
* Long-only, fully-invested portfolios with 5 bps round-trip costs.
* Core / Satellite / Non-Traditional / Opportunistic sleeve limits.

Only assets with local data are tradable in this repo. Missing Whitmore IPS
series are not fabricated; the code uses documented local proxies only where
the asset class is directly represented in `data/`.
"""

from __future__ import annotations

import pandas as pd

ANNUALIZATION = 252
RISK_FREE_RATE = 0.02
ROUND_TRIP_COST_BPS = 5.0
BASE_WINDOW = 252

# Zion OHLCV tickers with usable local price history.
PRICE_TICKERS = ["SPXT", "LBUSTRUU", "B3REITT", "XAU", "XBTUSD"]

# Available Whitmore / proxy tradable universe. `US TIPS` is proxied by the
# local `0_5Y_TIPS_2002_D` series because `BROAD_TIPS` is not included.
ASSET_ORDER = [
    "US Equity",
    "US Treasuries",
    "US TIPS",
    "US REITs",
    "Gold",
    "Bitcoin",
]

ASSET_LABELS = {
    "SPXT": "US Equity",
    "LBUSTRUU": "US Treasuries",
    "0_5Y_TIPS_2002_D": "US TIPS",
    "B3REITT": "US REITs",
    "XAU": "Gold",
    "XBTUSD": "Bitcoin",
}

ASSET_TICKERS = {
    "US Equity": "SPXT",
    "US Treasuries": "LBUSTRUU",
    "US TIPS": "0_5Y_TIPS_2002_D",
    "US REITs": "B3REITT",
    "Gold": "XAU",
    "Bitcoin": "XBTUSD",
}

ASSET_CLASSIFICATION = {
    "US Equity": "core",
    "US Treasuries": "core",
    "US TIPS": "core",
    "US REITs": "satellite",
    "Gold": "satellite",
    "Bitcoin": "non_traditional",
}

CORE_ASSETS = [asset for asset, group in ASSET_CLASSIFICATION.items() if group == "core"]
SATELLITE_ASSETS = [asset for asset, group in ASSET_CLASSIFICATION.items() if group == "satellite"]
NON_TRADITIONAL = [asset for asset, group in ASSET_CLASSIFICATION.items() if group == "non_traditional"]
OPPORTUNISTIC_ASSETS = [asset for asset, group in ASSET_CLASSIFICATION.items() if group == "opportunistic"]

RISKY_ASSETS = ["US Equity", "US REITs", "Bitcoin"]
DEFENSIVE_ASSETS = ["US Treasuries", "US TIPS", "Gold"]

CORE_FLOOR = 0.40
SATELLITE_CAP = 0.45
NON_TRADITIONAL_CAP = 0.20  # 2026 amendment supersedes the original 15% cap.
OPPORTUNISTIC_CAP = 0.15
OPPORTUNISTIC_SINGLE_ASSET_CAP = 0.05
SINGLE_SLEEVE_MAX = 0.45

SAA_TARGET_WEIGHTS = pd.Series(
    {
        "US Equity": 0.40,
        "US Treasuries": 0.10,
        "US TIPS": 0.05,
        "US REITs": 0.10,
        "Gold": 0.15,
        "Bitcoin": 0.00,
    },
    dtype=float,
)

SAA_LOWER_BOUNDS = pd.Series(
    {
        "US Equity": 0.30,
        "US Treasuries": 0.05,
        "US TIPS": 0.00,
        "US REITs": 0.05,
        "Gold": 0.10,
        "Bitcoin": 0.00,
    },
    dtype=float,
)

SAA_UPPER_BOUNDS = pd.Series(
    {
        "US Equity": 0.45,
        "US Treasuries": 0.15,
        "US TIPS": 0.10,
        "US REITs": 0.20,
        "Gold": 0.25,
        "Bitcoin": 0.05,
    },
    dtype=float,
)

TAA_LOWER_BOUNDS = pd.Series(
    {
        "US Equity": 0.20,
        "US Treasuries": 0.00,
        "US TIPS": 0.00,
        "US REITs": 0.00,
        "Gold": 0.00,
        "Bitcoin": 0.00,
    },
    dtype=float,
)

TAA_UPPER_BOUNDS = pd.Series(
    {
        "US Equity": 0.45,
        "US Treasuries": 0.35,
        "US TIPS": 0.25,
        "US REITs": 0.25,
        "Gold": 0.30,
        "Bitcoin": 0.10,
    },
    dtype=float,
)

# Backward-compatible aliases used by older helper code. SAA construction uses
# policy bounds by default; tactical code passes explicit TAA bounds.
LOWER_BOUNDS = SAA_LOWER_BOUNDS
UPPER_BOUNDS = SAA_UPPER_BOUNDS

BENCHMARK_1_WEIGHTS = pd.Series(
    {
        "US Equity": 0.60,
        "US Treasuries": 0.40,
    },
    dtype=float,
)

BENCHMARK_2_WEIGHTS = pd.Series(
    {
        "US Equity": 0.40,
        "Japan Equity": 0.05,
        "China A-Shares": 0.05,
        "US Treasuries": 0.10,
        "US TIPS": 0.05,
        "US REITs": 0.10,
        "Gold": 0.15,
        "Silver": 0.05,
        "Swiss Franc": 0.05,
    },
    dtype=float,
)

# Backward-compatible alias; Benchmark 1 is the traditional 60/40 policy.
BENCHMARK_WEIGHTS = BENCHMARK_1_WEIGHTS

BENCHMARK_DEFINITIONS = {
    "Benchmark 1 60/40": BENCHMARK_1_WEIGHTS,
    "Benchmark 2 Diversified Policy": BENCHMARK_2_WEIGHTS,
}

BENCHMARK_PROXY_NOTES = {
    "US TIPS": "BROAD_TIPS is unavailable; using local 0_5Y_TIPS_2002_D as the TIPS proxy.",
    "Japan Equity": "Unavailable in local data; dropped from computable Benchmark 2 weights.",
    "China A-Shares": "Unavailable in local data; dropped from computable Benchmark 2 weights.",
    "Silver": "Unavailable in local data; dropped from computable Benchmark 2 weights.",
    "Swiss Franc": "Unavailable in local data; dropped from computable Benchmark 2 weights.",
}

TAA_DEVIATION_LIMIT = 1.0
IPS_RETURN_TARGET = 0.08
IPS_VOL_TARGET = 0.15
IPS_MAX_DRAWDOWN_TARGET = -0.25


def available_benchmark_weights(weights: pd.Series, available_assets: list[str]) -> pd.Series:
    """Drop unavailable benchmark components and renormalize to fully invested."""
    available = weights.reindex(available_assets).dropna()
    available = available[available > 0]
    total = float(available.sum())
    if total <= 0:
        return pd.Series(dtype=float)
    return available / total


def ips_pass_count(metrics: pd.Series | dict) -> int:
    """Number of binding IPS constraints satisfied (return / vol / drawdown)."""
    return (
        int(float(metrics["total_return_pa"]) >= IPS_RETURN_TARGET)
        + int(float(metrics["volatility_pa"]) <= IPS_VOL_TARGET)
        + int(float(metrics["max_drawdown"]) >= IPS_MAX_DRAWDOWN_TARGET)
    )


def ips_loss(metrics: pd.Series | dict) -> float:
    """Backward-compatible alias for the current IPS objective."""
    return ips_objective(metrics)


def ips_constraint_gaps(metrics: pd.Series | dict) -> tuple[float, float, float]:
    """Return the miss-to-target for (return, vol, drawdown)."""
    return_miss = max(0.0, IPS_RETURN_TARGET - float(metrics["total_return_pa"]))
    vol_miss = max(0.0, float(metrics["volatility_pa"]) - IPS_VOL_TARGET)
    dd_miss = max(0.0, IPS_MAX_DRAWDOWN_TARGET - float(metrics["max_drawdown"]))
    return return_miss, vol_miss, dd_miss


def ips_objective(metrics: pd.Series | dict) -> float:
    """Constraint-first scalar ranking metric. Lower is better.

    Priority order:
    1. Maximize the number of satisfied IPS constraints (3/3 is the target).
    2. Minimize the distance to any unsatisfied constraints.
    3. Among portfolios with the same pass count and gap profile, prefer
       higher return, lower volatility, and shallower drawdown.
    """
    return_miss, vol_miss, dd_miss = ips_constraint_gaps(metrics)
    passes = ips_pass_count(metrics)
    total_return = float(metrics["total_return_pa"])
    total_vol = float(metrics["volatility_pa"])
    max_dd = float(metrics["max_drawdown"])
    pass_penalty = float((3 - passes) * 1_000.0)
    gap_penalty = float(100.0 * (return_miss + vol_miss + dd_miss))
    metric_tiebreak = float((-100.0 * total_return) + (100.0 * total_vol) - (100.0 * max_dd))
    return pass_penalty + gap_penalty + metric_tiebreak


def ips_scorecard(metrics: pd.Series | dict) -> pd.DataFrame:
    """Tabular scorecard showing target / achieved / gap / pass per constraint."""
    rows = [
        (
            "Return p.a. >= 8%",
            f">= {IPS_RETURN_TARGET:.1%}",
            float(metrics["total_return_pa"]),
            float(metrics["total_return_pa"]) - IPS_RETURN_TARGET,
            float(metrics["total_return_pa"]) >= IPS_RETURN_TARGET,
        ),
        (
            "Volatility p.a. <= 15%",
            f"<= {IPS_VOL_TARGET:.1%}",
            float(metrics["volatility_pa"]),
            IPS_VOL_TARGET - float(metrics["volatility_pa"]),
            float(metrics["volatility_pa"]) <= IPS_VOL_TARGET,
        ),
        (
            "Max drawdown >= -25%",
            f">= {IPS_MAX_DRAWDOWN_TARGET:.1%}",
            float(metrics["max_drawdown"]),
            float(metrics["max_drawdown"]) - IPS_MAX_DRAWDOWN_TARGET,
            float(metrics["max_drawdown"]) >= IPS_MAX_DRAWDOWN_TARGET,
        ),
    ]
    return pd.DataFrame(rows, columns=["constraint", "target", "achieved", "gap", "passed"])


__all__ = [
    "ANNUALIZATION",
    "ASSET_CLASSIFICATION",
    "ASSET_LABELS",
    "ASSET_ORDER",
    "ASSET_TICKERS",
    "BASE_WINDOW",
    "BENCHMARK_1_WEIGHTS",
    "BENCHMARK_2_WEIGHTS",
    "BENCHMARK_DEFINITIONS",
    "BENCHMARK_PROXY_NOTES",
    "BENCHMARK_WEIGHTS",
    "CORE_ASSETS",
    "CORE_FLOOR",
    "DEFENSIVE_ASSETS",
    "IPS_MAX_DRAWDOWN_TARGET",
    "IPS_RETURN_TARGET",
    "IPS_VOL_TARGET",
    "LOWER_BOUNDS",
    "NON_TRADITIONAL",
    "NON_TRADITIONAL_CAP",
    "OPPORTUNISTIC_ASSETS",
    "OPPORTUNISTIC_CAP",
    "OPPORTUNISTIC_SINGLE_ASSET_CAP",
    "PRICE_TICKERS",
    "RISK_FREE_RATE",
    "RISKY_ASSETS",
    "ROUND_TRIP_COST_BPS",
    "SATELLITE_ASSETS",
    "SATELLITE_CAP",
    "SAA_LOWER_BOUNDS",
    "SAA_TARGET_WEIGHTS",
    "SAA_UPPER_BOUNDS",
    "SINGLE_SLEEVE_MAX",
    "TAA_DEVIATION_LIMIT",
    "TAA_LOWER_BOUNDS",
    "TAA_UPPER_BOUNDS",
    "UPPER_BOUNDS",
    "available_benchmark_weights",
    "ips_loss",
    "ips_constraint_gaps",
    "ips_objective",
    "ips_pass_count",
    "ips_scorecard",
]
