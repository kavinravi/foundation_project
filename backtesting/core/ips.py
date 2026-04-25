"""Whitmore IPS constants and acceptance helpers.

Re-exports the binding constants from the legacy ``build_assignment_4.py`` so
that the new backtesting framework stays consistent with prior work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_assignment_4 import (  # noqa: E402
    ANNUALIZATION,
    ASSET_LABELS,
    ASSET_ORDER,
    BENCHMARK_WEIGHTS,
    DEFENSIVE_ASSETS,
    IPS_MAX_DRAWDOWN_TARGET,
    IPS_RETURN_TARGET,
    IPS_VOL_TARGET,
    LOWER_BOUNDS,
    NON_TRADITIONAL,
    NON_TRADITIONAL_CAP,
    PRICE_TICKERS,
    RISK_FREE_RATE,
    RISKY_ASSETS,
    ROUND_TRIP_COST_BPS,
    TAA_DEVIATION_LIMIT,
    UPPER_BOUNDS,
)


def ips_pass_count(metrics: pd.Series | dict) -> int:
    """Number of binding IPS constraints satisfied (return / vol / drawdown)."""
    return (
        int(metrics["total_return_pa"] >= IPS_RETURN_TARGET)
        + int(metrics["volatility_pa"] <= IPS_VOL_TARGET)
        + int(metrics["max_drawdown"] >= IPS_MAX_DRAWDOWN_TARGET)
    )


def ips_loss(metrics: pd.Series | dict) -> float:
    """Composite scalar objective. Lower is better.

    A feasible portfolio (all three IPS constraints met) has zero penalty
    and is then ordered by Sharpe. Infeasible portfolios pay a linear
    penalty proportional to how badly each constraint is missed.
    """
    return_miss = max(0.0, IPS_RETURN_TARGET - float(metrics["total_return_pa"]))
    vol_miss = max(0.0, float(metrics["volatility_pa"]) - IPS_VOL_TARGET)
    dd_miss = max(0.0, IPS_MAX_DRAWDOWN_TARGET - float(metrics["max_drawdown"]))
    penalty = 100.0 * (return_miss + vol_miss + dd_miss)
    sharpe = float(metrics.get("sharpe_rf2", 0.0) or 0.0)
    return penalty - 0.10 * sharpe


def ips_scorecard(metrics: pd.Series | dict) -> pd.DataFrame:
    """Tabular scorecard showing target / achieved / gap / pass per constraint."""
    rows = [
        (
            "Return p.a. ≥ 5%",
            f"≥ {IPS_RETURN_TARGET:.1%}",
            float(metrics["total_return_pa"]),
            float(metrics["total_return_pa"]) - IPS_RETURN_TARGET,
            float(metrics["total_return_pa"]) >= IPS_RETURN_TARGET,
        ),
        (
            "Volatility p.a. ≤ 11%",
            f"≤ {IPS_VOL_TARGET:.1%}",
            float(metrics["volatility_pa"]),
            IPS_VOL_TARGET - float(metrics["volatility_pa"]),
            float(metrics["volatility_pa"]) <= IPS_VOL_TARGET,
        ),
        (
            "Max drawdown ≥ -13%",
            f"≥ {IPS_MAX_DRAWDOWN_TARGET:.1%}",
            float(metrics["max_drawdown"]),
            float(metrics["max_drawdown"]) - IPS_MAX_DRAWDOWN_TARGET,
            float(metrics["max_drawdown"]) >= IPS_MAX_DRAWDOWN_TARGET,
        ),
    ]
    return pd.DataFrame(rows, columns=["constraint", "target", "achieved", "gap", "passed"])


__all__ = [
    "ANNUALIZATION",
    "ASSET_LABELS",
    "ASSET_ORDER",
    "BENCHMARK_WEIGHTS",
    "DEFENSIVE_ASSETS",
    "IPS_MAX_DRAWDOWN_TARGET",
    "IPS_RETURN_TARGET",
    "IPS_VOL_TARGET",
    "LOWER_BOUNDS",
    "NON_TRADITIONAL",
    "NON_TRADITIONAL_CAP",
    "PRICE_TICKERS",
    "RISK_FREE_RATE",
    "RISKY_ASSETS",
    "ROUND_TRIP_COST_BPS",
    "TAA_DEVIATION_LIMIT",
    "UPPER_BOUNDS",
    "ips_loss",
    "ips_pass_count",
    "ips_scorecard",
]
