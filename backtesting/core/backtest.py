"""Backtest engine wrappers.

Wraps the SAA optimizer, walk-forward folding, and TAA overlay logic
from the legacy ``build_assignment_4.py`` so that hypothesis-specific
modules can stay short and declarative.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_assignment_4 import (  # noqa: E402
    BASE_WINDOW,
    BacktestResult,
    annual_rebalance_dates,
    apply_taa_overlay,
    base_weight_for_date,
    build_minvar_targets,
    compute_metrics,
    feasible_for_assets,
    first_feasible_date,
    first_full_overlap_date,
    grouped_period_end_dates,
    history_slice,
    run_strategy_backtest,
    solve_min_variance_with_bounds,
    solve_target_return_frontier,
)

from backtesting.core.ips import (  # noqa: E402
    ASSET_ORDER,
    BENCHMARK_WEIGHTS,
    IPS_RETURN_TARGET,
    LOWER_BOUNDS,
    NON_TRADITIONAL,
    NON_TRADITIONAL_CAP,
    TAA_DEVIATION_LIMIT,
    UPPER_BOUNDS,
)


# ---------------------------------------------------------------------------
# SAA construction
# ---------------------------------------------------------------------------


def solve_ips_min_variance(
    covariance: pd.DataFrame,
    active_assets: list[str],
    target_return: float = IPS_RETURN_TARGET,
    expected_returns: pd.Series | None = None,
) -> pd.Series:
    """Min-variance solve subject to all binding IPS bounds + non-trad cap.

    If ``expected_returns`` is provided, also enforces ``w·μ ≥ target_return``.
    """
    lower = LOWER_BOUNDS.reindex(active_assets).fillna(0.0)
    upper = UPPER_BOUNDS.reindex(active_assets).fillna(0.0)
    if expected_returns is not None:
        try:
            return solve_target_return_frontier(
                covariance=covariance,
                expected_returns=expected_returns.reindex(active_assets),
                active_assets=active_assets,
                target_return=target_return,
            )
        except Exception:
            pass
    return solve_min_variance_with_bounds(
        covariance=covariance,
        active_assets=active_assets,
        lower_bounds=lower,
        upper_bounds=upper,
    )


def feasible_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Trim the returns frame to the first IPS-feasible rebalance date.

    "Feasible" here means the active asset universe at the point-in-time
    rebalance can satisfy the lower bounds + non-traditional cap defined
    in the IPS.
    """
    filled = returns.add(1.0).fillna(1.0).cumprod()
    start = first_feasible_date(filled)
    return returns.loc[start:]


def build_saa_targets(
    returns: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    covariance_window: int | None = BASE_WINDOW,
) -> dict[pd.Timestamp, pd.Series]:
    """Annual SAA weights via IPS-constrained minimum variance.

    Skips rebalance dates where the active universe is infeasible (e.g. before
    Bitcoin and REITs both have data, the lower bounds cannot be met).
    """
    targets: dict[pd.Timestamp, pd.Series] = {}
    for as_of in rebalance_dates:
        active = [
            asset for asset in ASSET_ORDER if returns.loc[:as_of, asset].notna().any()
        ]
        if not feasible_for_assets(active):
            continue
        history = history_slice(returns[active].dropna(how="all"), as_of, covariance_window)
        cov = history.cov(min_periods=max(20, min(len(history), 20)))
        try:
            targets[as_of] = solve_ips_min_variance(cov, active)
        except RuntimeError:
            continue
    return targets


def run_saa_backtest(
    returns: pd.DataFrame,
    covariance_window: int | None = BASE_WINDOW,
    strategy_name: str = "SAA",
) -> BacktestResult:
    returns = feasible_returns(returns)
    rebalance_dates = annual_rebalance_dates(returns.index)
    targets = build_saa_targets(returns, rebalance_dates, covariance_window=covariance_window)
    if not targets:
        raise RuntimeError("No feasible SAA rebalance dates in the supplied returns frame.")
    first_rebalance = min(targets.keys())
    returns_for_run = returns.loc[first_rebalance:]
    return run_strategy_backtest(
        returns=returns_for_run, target_weights_by_date=targets, strategy_name=strategy_name
    )


def run_benchmark_backtest(
    returns: pd.DataFrame, strategy_name: str = "Benchmark 60/40"
) -> BacktestResult:
    returns = feasible_returns(returns)
    rebalance_dates = annual_rebalance_dates(returns.index)
    rebalance_dates = [d for d in rebalance_dates if returns.loc[:d, "US Equity"].notna().any()]
    targets = {date: BENCHMARK_WEIGHTS.copy() for date in rebalance_dates}
    return run_strategy_backtest(returns=returns, target_weights_by_date=targets, strategy_name=strategy_name)


# ---------------------------------------------------------------------------
# TAA overlay driven by an arbitrary signal series
# ---------------------------------------------------------------------------


def signal_to_taa_targets(
    signals: pd.Series,
    saa_result: BacktestResult,
    returns: pd.DataFrame,
    *,
    decision_dates: list[pd.Timestamp] | None = None,
    deviation_limit: float = TAA_DEVIATION_LIMIT,
    covariance_window: int | None = BASE_WINDOW,
    risk_off_tilt: dict[str, float] | None = None,
    risk_on_tilt: dict[str, float] | None = None,
) -> dict[pd.Timestamp, pd.Series]:
    """Map a {-1, 0, +1} signal series into TAA target weights.

    ``signal == -1``  → risk-off: shrink risk-asset weights by ``deviation_limit``,
                        rebalance the freed weight into the ``risk_off_tilt``
                        assets (within IPS bands).
    ``signal == 0``   → keep SAA weights.
    ``signal == +1``  → risk-on: tilt into the ``risk_on_tilt`` assets up to
                        ``deviation_limit`` per asset (within IPS bands).

    If no tilt dict is provided the function falls back to the legacy
    ``apply_taa_overlay`` (covariance-aware min-variance within ±15% bands).
    """
    if decision_dates is None:
        decision_dates = list(signals.dropna().index)

    targets: dict[pd.Timestamp, pd.Series] = {}
    for date in decision_dates:
        date = pd.Timestamp(date)
        signal_value = float(signals.loc[:date].iloc[-1]) if not signals.loc[:date].empty else 0.0
        base_w = base_weight_for_date(saa_result.weights_by_rebalance, date)

        if signal_value == 0 or (risk_off_tilt is None and risk_on_tilt is None):
            if signal_value == 0:
                targets[date] = base_w.copy()
            else:
                targets[date] = apply_taa_overlay(
                    base_weights=base_w,
                    returns=returns,
                    decision_date=date,
                    covariance_window=covariance_window,
                    deviation_limit=deviation_limit,
                )
            continue

        tilt = risk_off_tilt if signal_value < 0 else risk_on_tilt
        targets[date] = _apply_directional_tilt(base_w, tilt or {}, deviation_limit)

    return targets


def _apply_directional_tilt(
    base_weights: pd.Series,
    tilt: dict[str, float],
    deviation_limit: float,
) -> pd.Series:
    """Apply a per-asset additive tilt clipped to ±deviation_limit and IPS bands.

    The tilt dict expresses *desired* per-asset adjustments in weight space.
    They are scaled if needed so the post-tilt weights still respect the IPS
    bounds (per-asset upper/lower, non-traditional cap) and sum to 1.
    """
    weights = base_weights.reindex(ASSET_ORDER).fillna(0.0).copy()
    raw_tilt = pd.Series({asset: tilt.get(asset, 0.0) for asset in ASSET_ORDER})

    raw_tilt = raw_tilt.clip(lower=-deviation_limit, upper=deviation_limit)
    if raw_tilt.sum() != 0:
        raw_tilt = raw_tilt - raw_tilt.sum() / len(raw_tilt)

    new_weights = (weights + raw_tilt).clip(lower=LOWER_BOUNDS, upper=UPPER_BOUNDS)
    new_weights = new_weights.clip(
        lower=(weights - deviation_limit).clip(lower=0.0),
        upper=(weights + deviation_limit).clip(upper=1.0),
    )
    new_weights = _enforce_non_traditional_cap(new_weights)
    if new_weights.sum() > 0:
        new_weights = new_weights / new_weights.sum()
    return new_weights


def _enforce_non_traditional_cap(weights: pd.Series) -> pd.Series:
    nontrad = [a for a in NON_TRADITIONAL if a in weights.index]
    excess = float(weights[nontrad].sum()) - NON_TRADITIONAL_CAP
    if excess <= 0:
        return weights
    weights = weights.copy()
    weights[nontrad] = weights[nontrad] * (NON_TRADITIONAL_CAP / weights[nontrad].sum())
    return weights


def _snap_to_index(date: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Snap to the last index date ≤ ``date``. Returns None if no such date."""
    eligible = index[index <= date]
    if len(eligible) == 0:
        return None
    return pd.Timestamp(eligible[-1])


def run_taa_backtest(
    signals: pd.Series,
    saa_result: BacktestResult,
    returns: pd.DataFrame,
    *,
    decision_dates: list[pd.Timestamp] | None = None,
    deviation_limit: float = TAA_DEVIATION_LIMIT,
    covariance_window: int | None = BASE_WINDOW,
    risk_off_tilt: dict[str, float] | None = None,
    risk_on_tilt: dict[str, float] | None = None,
    strategy_name: str = "SAA + TAA",
) -> BacktestResult:
    if decision_dates is None:
        decision_dates = grouped_period_end_dates(returns.index, "monthly")

    snapped_pairs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for date in decision_dates:
        snapped = _snap_to_index(pd.Timestamp(date), returns.index)
        if snapped is not None:
            snapped_pairs.append((pd.Timestamp(date), snapped))
    if not snapped_pairs:
        raise RuntimeError("No decision dates fell within the returns index.")

    snapped_signals = pd.Series(
        {snap: float(signals.loc[orig]) for orig, snap in snapped_pairs if orig in signals.index}
    ).sort_index()

    snapped_signals = snapped_signals[~snapped_signals.index.duplicated(keep="last")]
    snapped_decision_dates = list(snapped_signals.index)

    saa_targets = saa_result.weights_by_rebalance.copy()
    signal_targets = signal_to_taa_targets(
        signals=snapped_signals,
        saa_result=saa_result,
        returns=returns,
        decision_dates=snapped_decision_dates,
        deviation_limit=deviation_limit,
        covariance_window=covariance_window,
        risk_off_tilt=risk_off_tilt,
        risk_on_tilt=risk_on_tilt,
    )

    combined: dict[pd.Timestamp, pd.Series] = {}
    for date, w in saa_targets.iterrows():
        snapped = _snap_to_index(pd.Timestamp(date), returns.index)
        if snapped is not None:
            combined[snapped] = w
    for date, w in signal_targets.items():
        combined[pd.Timestamp(date)] = w
    combined = dict(sorted(combined.items()))
    return run_strategy_backtest(
        returns=returns, target_weights_by_date=combined, strategy_name=strategy_name
    )


__all__ = [
    "BacktestResult",
    "build_saa_targets",
    "run_benchmark_backtest",
    "run_saa_backtest",
    "run_taa_backtest",
    "signal_to_taa_targets",
    "solve_ips_min_variance",
]
