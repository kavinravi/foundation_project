"""Backtest engine wrappers and configurable portfolio utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from backtesting.core.ips import (
    ANNUALIZATION,
    ASSET_ORDER,
    ASSET_CLASSIFICATION,
    BASE_WINDOW,
    BENCHMARK_1_WEIGHTS,
    CORE_ASSETS,
    CORE_FLOOR,
    IPS_RETURN_TARGET,
    NON_TRADITIONAL,
    NON_TRADITIONAL_CAP,
    OPPORTUNISTIC_ASSETS,
    OPPORTUNISTIC_CAP,
    OPPORTUNISTIC_SINGLE_ASSET_CAP,
    RISK_FREE_RATE,
    ROUND_TRIP_COST_BPS,
    SATELLITE_ASSETS,
    SATELLITE_CAP,
    SAA_LOWER_BOUNDS,
    SAA_UPPER_BOUNDS,
    SINGLE_SLEEVE_MAX,
    TAA_DEVIATION_LIMIT,
    TAA_LOWER_BOUNDS,
    TAA_UPPER_BOUNDS,
    available_benchmark_weights,
)


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    weights_by_rebalance: pd.DataFrame
    turnover_by_rebalance: pd.Series
    metrics: pd.Series


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


def annual_rebalance_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    return grouped_period_end_dates(index=index, frequency="annual", drop_last=True)


def history_slice(returns: pd.DataFrame, as_of: pd.Timestamp, window: int | None) -> pd.DataFrame:
    history = returns.loc[:as_of]
    if window is not None:
        history = history.tail(min(window, len(history)))
    return history


def _resolve_asset_universe(
    columns: pd.Index | list[str],
    asset_universe: list[str] | None = None,
) -> list[str]:
    columns = list(columns)
    if asset_universe is None:
        return [asset for asset in ASSET_ORDER if asset in columns]
    return [asset for asset in asset_universe if asset in columns]


def _asset_indices(active_assets: list[str], group_assets: list[str]) -> list[int]:
    group = set(group_assets)
    return [idx for idx, asset in enumerate(active_assets) if asset in group]


def _sanitize_bounds(
    active_assets: list[str],
    lower_bounds: pd.Series,
    upper_bounds: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    lower = lower_bounds.reindex(active_assets).fillna(0.0).astype(float)
    upper = upper_bounds.reindex(active_assets).fillna(1.0).astype(float)
    upper = upper.clip(upper=SINGLE_SLEEVE_MAX)
    for asset in active_assets:
        if ASSET_CLASSIFICATION.get(asset) == "opportunistic":
            upper.loc[asset] = min(float(upper.loc[asset]), OPPORTUNISTIC_SINGLE_ASSET_CAP)
    if (lower > upper + 1e-12).any():
        bad = lower.index[lower > upper + 1e-12].tolist()
        raise RuntimeError(f"Infeasible bounds: lower exceeds upper for {bad}")
    return lower, upper


def _linear_constraint_matrices(active_assets: list[str]) -> tuple[np.ndarray | None, np.ndarray | None]:
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    core_idx = _asset_indices(active_assets, CORE_ASSETS)
    sat_idx = _asset_indices(active_assets, SATELLITE_ASSETS)
    nontrad_idx = _asset_indices(active_assets, NON_TRADITIONAL)
    opp_idx = _asset_indices(active_assets, OPPORTUNISTIC_ASSETS)

    if CORE_FLOOR > 0:
        row = np.zeros(len(active_assets), dtype=float)
        if core_idx:
            row[core_idx] = -1.0
        rows.append(row)
        rhs.append(-CORE_FLOOR)
    if sat_idx and SATELLITE_CAP < 1.0:
        row = np.zeros(len(active_assets), dtype=float)
        row[sat_idx] = 1.0
        rows.append(row)
        rhs.append(SATELLITE_CAP)
    if nontrad_idx and NON_TRADITIONAL_CAP < 1.0:
        row = np.zeros(len(active_assets), dtype=float)
        row[nontrad_idx] = 1.0
        rows.append(row)
        rhs.append(NON_TRADITIONAL_CAP)
    if opp_idx and OPPORTUNISTIC_CAP < 1.0:
        row = np.zeros(len(active_assets), dtype=float)
        row[opp_idx] = 1.0
        rows.append(row)
        rhs.append(OPPORTUNISTIC_CAP)

    if not rows:
        return None, None
    return np.vstack(rows), np.asarray(rhs, dtype=float)


def _optimizer_constraints(active_assets: list[str]) -> list[dict]:
    constraints: list[dict] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    core_idx = _asset_indices(active_assets, CORE_ASSETS)
    sat_idx = _asset_indices(active_assets, SATELLITE_ASSETS)
    nontrad_idx = _asset_indices(active_assets, NON_TRADITIONAL)
    opp_idx = _asset_indices(active_assets, OPPORTUNISTIC_ASSETS)

    if CORE_FLOOR > 0:
        constraints.append(
            {"type": "ineq", "fun": lambda w, idx=core_idx: np.sum(w[idx]) - CORE_FLOOR}
        )
    if sat_idx and SATELLITE_CAP < 1.0:
        constraints.append(
            {"type": "ineq", "fun": lambda w, idx=sat_idx: SATELLITE_CAP - np.sum(w[idx])}
        )
    if nontrad_idx and NON_TRADITIONAL_CAP < 1.0:
        constraints.append(
            {"type": "ineq", "fun": lambda w, idx=nontrad_idx: NON_TRADITIONAL_CAP - np.sum(w[idx])}
        )
    if opp_idx and OPPORTUNISTIC_CAP < 1.0:
        constraints.append(
            {"type": "ineq", "fun": lambda w, idx=opp_idx: OPPORTUNISTIC_CAP - np.sum(w[idx])}
        )
    return constraints


def feasible_for_assets(
    active_assets: list[str],
    lower_bounds: pd.Series | None = None,
    upper_bounds: pd.Series | None = None,
) -> bool:
    if not active_assets:
        return False
    lower_bounds = SAA_LOWER_BOUNDS if lower_bounds is None else lower_bounds
    upper_bounds = SAA_UPPER_BOUNDS if upper_bounds is None else upper_bounds
    try:
        lower, upper = _sanitize_bounds(active_assets, lower_bounds, upper_bounds)
    except RuntimeError:
        return False
    if float(lower.sum()) > 1.0 + 1e-9 or float(upper.sum()) < 1.0 - 1e-9:
        return False
    a_ub, b_ub = _linear_constraint_matrices(active_assets)
    result = linprog(
        c=np.zeros(len(active_assets), dtype=float),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=np.ones((1, len(active_assets)), dtype=float),
        b_eq=np.array([1.0], dtype=float),
        bounds=list(zip(lower.to_numpy(dtype=float), upper.to_numpy(dtype=float))),
        method="highs",
    )
    return bool(result.success)


def first_feasible_date(
    filled_prices: pd.DataFrame,
    asset_universe: list[str] | None = None,
) -> pd.Timestamp:
    assets = _resolve_asset_universe(filled_prices.columns, asset_universe)
    start_dates = {asset: filled_prices[asset].dropna().index.min() for asset in assets}
    candidate_dates = sorted(set(start_dates.values()))
    for date in candidate_dates:
        active_assets = [asset for asset in assets if start_dates[asset] <= date]
        if feasible_for_assets(active_assets):
            return date
    raise RuntimeError("Could not find a feasible portfolio start date.")


def _initial_guess(
    active_assets: list[str],
    lower_bounds: pd.Series,
    upper_bounds: pd.Series,
) -> np.ndarray:
    a_ub, b_ub = _linear_constraint_matrices(active_assets)
    result = linprog(
        c=np.zeros(len(active_assets), dtype=float),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=np.ones((1, len(active_assets)), dtype=float),
        b_eq=np.array([1.0], dtype=float),
        bounds=list(zip(lower_bounds.to_numpy(dtype=float), upper_bounds.to_numpy(dtype=float))),
        method="highs",
    )
    if result.success:
        return result.x

    guess = lower_bounds.copy()
    slack = 1.0 - float(guess.sum())
    room = upper_bounds - lower_bounds
    if slack > 0 and float(room.sum()) > 0:
        guess += slack * room / float(room.sum())
    return guess.to_numpy(dtype=float)


def _non_trad_indices(active_assets: list[str]) -> list[int]:
    return _asset_indices(active_assets, NON_TRADITIONAL)


def solve_min_variance_with_bounds(
    covariance: pd.DataFrame,
    active_assets: list[str],
    lower_bounds: pd.Series,
    upper_bounds: pd.Series,
) -> pd.Series:
    lower, upper = _sanitize_bounds(active_assets, lower_bounds, upper_bounds)
    if not feasible_for_assets(active_assets, lower, upper):
        raise RuntimeError("Infeasible tactical bounds for min-variance overlay.")

    active_cov = covariance.loc[active_assets, active_assets].fillna(0.0)
    cov = active_cov.to_numpy(dtype=float)
    x0 = _initial_guess(active_assets, lower, upper)
    constraints = _optimizer_constraints(active_assets)

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

    weights = pd.Series(0.0, index=active_assets, dtype=float)
    weights.loc[active_assets] = result.x
    weights[np.isclose(weights, 0.0, atol=1e-10)] = 0.0
    return weights


def solve_target_return_frontier(
    covariance: pd.DataFrame,
    expected_returns: pd.Series,
    target_return: float,
    active_assets: list[str],
) -> pd.Series | None:
    if not feasible_for_assets(active_assets, SAA_LOWER_BOUNDS, SAA_UPPER_BOUNDS):
        return None

    mu = expected_returns.reindex(active_assets).astype(float).to_numpy()
    cov = covariance.loc[active_assets, active_assets].fillna(0.0).to_numpy(dtype=float)
    lower, upper = _sanitize_bounds(active_assets, SAA_LOWER_BOUNDS, SAA_UPPER_BOUNDS)
    x0 = _initial_guess(active_assets, lower, upper)

    constraints = _optimizer_constraints(active_assets)
    constraints.append({"type": "eq", "fun": lambda w: float(w @ mu) - target_return})

    result = minimize(
        lambda w: float(w @ cov @ w),
        x0=x0,
        method="SLSQP",
        bounds=list(zip(lower.to_numpy(dtype=float), upper.to_numpy(dtype=float))),
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        return None

    weights = pd.Series(0.0, index=active_assets, dtype=float)
    weights.loc[active_assets] = result.x
    weights[np.isclose(weights, 0.0, atol=1e-10)] = 0.0
    return weights


def solve_mean_variance_with_bounds(
    covariance: pd.DataFrame,
    expected_returns: pd.Series,
    active_assets: list[str],
    lower_bounds: pd.Series,
    upper_bounds: pd.Series,
    risk_aversion: float = 4.0,
) -> pd.Series:
    """Long-only Markowitz optimizer using expected returns and covariance."""
    lower, upper = _sanitize_bounds(active_assets, lower_bounds, upper_bounds)
    if not feasible_for_assets(active_assets, lower, upper):
        raise RuntimeError("Infeasible tactical bounds for mean-variance optimizer.")

    cov_frame = covariance.loc[active_assets, active_assets].fillna(0.0)
    cov = cov_frame.to_numpy(dtype=float)
    cov = cov + np.eye(len(active_assets)) * 1e-8
    mu = expected_returns.reindex(active_assets).fillna(0.0).to_numpy(dtype=float)
    x0 = _initial_guess(active_assets, lower, upper)
    constraints = _optimizer_constraints(active_assets)

    def objective(w: np.ndarray) -> float:
        variance_penalty = 0.5 * risk_aversion * float(w @ cov @ w)
        expected_reward = float(w @ mu)
        return variance_penalty - expected_reward

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=list(zip(lower.to_numpy(dtype=float), upper.to_numpy(dtype=float))),
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Mean-variance optimization failed: {result.message}")

    weights = pd.Series(result.x, index=active_assets, dtype=float)
    weights[np.isclose(weights, 0.0, atol=1e-10)] = 0.0
    return weights


def project_weights_to_feasible(
    desired_weights: pd.Series,
    active_assets: list[str],
    lower_bounds: pd.Series,
    upper_bounds: pd.Series,
) -> pd.Series:
    """Nearest feasible IPS-compliant portfolio to ``desired_weights``."""
    lower, upper = _sanitize_bounds(active_assets, lower_bounds, upper_bounds)
    if not feasible_for_assets(active_assets, lower, upper):
        raise RuntimeError("Infeasible tactical bounds for weight projection.")

    desired = desired_weights.reindex(active_assets).fillna(0.0).to_numpy(dtype=float)
    x0 = _initial_guess(active_assets, lower, upper)
    constraints = _optimizer_constraints(active_assets)
    result = minimize(
        lambda w: float(np.square(w - desired).sum()),
        x0=x0,
        method="SLSQP",
        bounds=list(zip(lower.to_numpy(dtype=float), upper.to_numpy(dtype=float))),
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Weight projection failed: {result.message}")
    weights = pd.Series(result.x, index=active_assets, dtype=float)
    weights[np.isclose(weights, 0.0, atol=1e-10)] = 0.0
    return weights


def black_litterman_posterior(
    covariance: pd.DataFrame,
    market_weights: pd.Series,
    views: pd.Series,
    *,
    tau: float = 0.05,
    confidence: float = 0.50,
    risk_aversion: float = 4.0,
) -> pd.Series:
    """Blend equilibrium returns with absolute views using Black-Litterman.

    ``covariance`` is annualized. ``views`` are annualized expected returns.
    """
    assets = list(covariance.columns)
    cov = covariance.loc[assets, assets].fillna(0.0).to_numpy(dtype=float)
    cov = cov + np.eye(len(assets)) * 1e-8
    weights = market_weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    if weights.sum() <= 0:
        weights = np.repeat(1.0 / len(assets), len(assets))
    else:
        weights = weights / weights.sum()

    pi = risk_aversion * cov @ weights
    q = views.reindex(assets).fillna(pd.Series(pi, index=assets)).to_numpy(dtype=float)
    tau_cov = tau * cov
    conf = float(np.clip(confidence, 0.05, 0.99))
    omega_diag = np.maximum(np.diag(tau_cov), 1e-8) * (1.0 - conf) / conf
    omega = np.diag(omega_diag)

    try:
        lhs = np.linalg.inv(tau_cov) + np.linalg.inv(omega)
        rhs = np.linalg.inv(tau_cov) @ pi + np.linalg.inv(omega) @ q
        posterior = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        posterior = (1.0 - conf) * pi + conf * q
    return pd.Series(posterior, index=assets, dtype=float)


def solve_ips_min_variance(
    covariance: pd.DataFrame,
    active_assets: list[str],
    target_return: float = IPS_RETURN_TARGET,
    expected_returns: pd.Series | None = None,
) -> pd.Series:
    """Min-variance solve with a return floor when feasible."""
    lower = SAA_LOWER_BOUNDS.reindex(active_assets).fillna(0.0)
    upper = SAA_UPPER_BOUNDS.reindex(active_assets).fillna(1.0)
    if expected_returns is not None:
        try:
            weights = solve_target_return_frontier(
                covariance=covariance,
                expected_returns=expected_returns.reindex(active_assets),
                active_assets=active_assets,
                target_return=target_return,
            )
            if weights is not None:
                return weights
        except Exception:
            pass
    return solve_min_variance_with_bounds(
        covariance=covariance,
        active_assets=active_assets,
        lower_bounds=lower,
        upper_bounds=upper,
    )


def feasible_returns(
    returns: pd.DataFrame,
    asset_universe: list[str] | None = None,
) -> pd.DataFrame:
    """Trim the returns frame to the first feasible date for the chosen assets."""
    assets = _resolve_asset_universe(returns.columns, asset_universe)
    returns = returns.loc[:, assets].copy()
    started = returns.notna().cummax()
    filled = returns.where(started).fillna(0.0).add(1.0).cumprod().where(started)
    start = first_feasible_date(filled, asset_universe=assets)
    return returns.loc[start:]


def build_saa_targets(
    returns: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    covariance_window: int | None = BASE_WINDOW,
    asset_universe: list[str] | None = None,
) -> dict[pd.Timestamp, pd.Series]:
    assets = _resolve_asset_universe(returns.columns, asset_universe)
    targets: dict[pd.Timestamp, pd.Series] = {}
    for as_of in rebalance_dates:
        active = [asset for asset in assets if returns.loc[:as_of, asset].notna().any()]
        if not feasible_for_assets(active):
            continue
        history = history_slice(returns[active].dropna(how="all"), as_of, covariance_window)
        if len(history) < 20:
            continue
        cov = history.cov(min_periods=max(20, min(len(history), 20)))
        mu = history.mean() * ANNUALIZATION
        try:
            targets[as_of] = solve_ips_min_variance(cov, active, expected_returns=mu)
        except RuntimeError:
            continue
    return targets


def turnover_to_target(current_weights: pd.Series, target_weights: pd.Series) -> float:
    return 0.5 * float((target_weights - current_weights).abs().sum())


def drifted_weights(start_weights: pd.Series, period_returns: pd.DataFrame) -> pd.Series:
    if period_returns.empty:
        return start_weights.copy()
    filled_period = period_returns.fillna(0.0)
    growth = (1.0 + filled_period).prod()
    end_weights = start_weights * growth.reindex(start_weights.index).fillna(1.0)
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
    asset_columns = list(returns.columns)
    rebalance_dates = list(target_weights_by_date.keys())
    current_weights = pd.Series(0.0, index=asset_columns, dtype=float)
    daily_segments: list[pd.Series] = []
    turnover_records: list[tuple[pd.Timestamp, float]] = []

    for idx, rebalance_date in enumerate(rebalance_dates):
        target_weights = target_weights_by_date[rebalance_date].reindex(asset_columns).fillna(0.0)
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

    daily_returns = pd.concat(daily_segments).sort_index() if daily_segments else pd.Series(dtype=float)
    turnover_series = pd.Series(
        {timestamp: value for timestamp, value in turnover_records},
        name="turnover",
        dtype=float,
    ).sort_index()
    weights_frame = (
        pd.DataFrame(target_weights_by_date).T.reindex(columns=asset_columns).sort_index().rename_axis("rebalance_date")
    )
    metrics = compute_metrics(daily_returns)
    return BacktestResult(
        daily_returns=daily_returns,
        weights_by_rebalance=weights_frame,
        turnover_by_rebalance=turnover_series,
        metrics=metrics,
    )


def run_saa_backtest(
    returns: pd.DataFrame,
    covariance_window: int | None = BASE_WINDOW,
    strategy_name: str = "SAA",
    asset_universe: list[str] | None = None,
    run_start: pd.Timestamp | None = None,
) -> BacktestResult:
    returns = feasible_returns(returns, asset_universe=asset_universe)
    rebalance_dates = annual_rebalance_dates(returns.index)
    targets = build_saa_targets(
        returns,
        rebalance_dates,
        covariance_window=covariance_window,
        asset_universe=list(returns.columns),
    )
    if not targets:
        raise RuntimeError("No feasible SAA rebalance dates in the supplied returns frame.")
    if run_start is None:
        first_rebalance = min(targets.keys())
        returns_for_run = returns.loc[first_rebalance:]
        targets_for_run = targets
    else:
        run_start = pd.Timestamp(run_start)
        eligible = returns.index[returns.index <= run_start]
        if len(eligible) > 0:
            first_rebalance = pd.Timestamp(eligible[-1])
        else:
            future = returns.index[returns.index >= run_start]
            if len(future) == 0:
                raise RuntimeError("No return observations overlap the requested SAA run_start.")
            first_rebalance = pd.Timestamp(future[0])
        weights_frame = pd.DataFrame(targets).T.sort_index()
        if weights_frame.loc[weights_frame.index <= first_rebalance].empty:
            raise RuntimeError("No SAA target exists on or before the requested OOS start.")
        initial_target = base_weight_for_date(weights_frame, first_rebalance)
        targets_for_run = {first_rebalance: initial_target}
        for date, weights in sorted(targets.items()):
            if pd.Timestamp(date) > first_rebalance:
                targets_for_run[pd.Timestamp(date)] = weights
        returns_for_run = returns.loc[first_rebalance:]
    return run_strategy_backtest(
        returns=returns_for_run,
        target_weights_by_date=targets_for_run,
        strategy_name=strategy_name,
    )


def run_benchmark_backtest(
    returns: pd.DataFrame,
    strategy_name: str = "Benchmark 1 60/40",
    asset_universe: list[str] | None = None,
    benchmark_weights: pd.Series | None = None,
    run_start: pd.Timestamp | None = None,
) -> BacktestResult:
    returns = feasible_returns(returns, asset_universe=asset_universe)
    rebalance_dates = annual_rebalance_dates(returns.index)
    rebalance_dates = [d for d in rebalance_dates if returns.loc[:d, "US Equity"].notna().any()]
    raw_weights = BENCHMARK_1_WEIGHTS if benchmark_weights is None else benchmark_weights
    available_weights = available_benchmark_weights(raw_weights, list(returns.columns))
    if available_weights.empty:
        raise RuntimeError(f"No benchmark components are available for {strategy_name}.")
    targets = {date: available_weights.reindex(returns.columns).fillna(0.0) for date in rebalance_dates}
    if run_start is not None:
        run_start = pd.Timestamp(run_start)
        eligible = returns.index[returns.index <= run_start]
        if len(eligible) > 0:
            first_rebalance = pd.Timestamp(eligible[-1])
        else:
            future = returns.index[returns.index >= run_start]
            if len(future) == 0:
                raise RuntimeError("No return observations overlap the requested benchmark run_start.")
            first_rebalance = pd.Timestamp(future[0])
        targets = {date: weights for date, weights in targets.items() if pd.Timestamp(date) > first_rebalance}
        targets[first_rebalance] = available_weights.reindex(returns.columns).fillna(0.0)
        targets = dict(sorted(targets.items()))
        returns = returns.loc[first_rebalance:]
    return run_strategy_backtest(returns=returns, target_weights_by_date=targets, strategy_name=strategy_name)


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
    """Covariance-aware tactical overlay within the configured deviation band."""
    asset_columns = list(base_weights.index)
    active_assets = [asset for asset in asset_columns if returns.loc[:decision_date, asset].notna().any()]
    history = history_slice(returns[active_assets].dropna(how="all"), decision_date, covariance_window)
    covariance = history.cov(min_periods=max(20, min(len(history), 20)))
    lower_bounds = (base_weights.reindex(asset_columns).fillna(0.0) - deviation_limit).clip(lower=0.0)
    lower_bounds = lower_bounds.clip(lower=TAA_LOWER_BOUNDS.reindex(asset_columns).fillna(0.0))
    upper_bounds = (base_weights.reindex(asset_columns).fillna(0.0) + deviation_limit).clip(upper=1.0)
    upper_bounds = upper_bounds.clip(upper=TAA_UPPER_BOUNDS.reindex(asset_columns).fillna(1.0))
    return solve_min_variance_with_bounds(covariance, active_assets, lower_bounds, upper_bounds)


def _enforce_non_traditional_cap(weights: pd.Series) -> pd.Series:
    if NON_TRADITIONAL_CAP >= 1.0:
        return weights
    nontrad = [asset for asset in NON_TRADITIONAL if asset in weights.index]
    excess = float(weights[nontrad].sum()) - NON_TRADITIONAL_CAP
    if excess <= 0:
        return weights
    weights = weights.copy()
    weights[nontrad] = weights[nontrad] * (NON_TRADITIONAL_CAP / weights[nontrad].sum())
    return weights


def _apply_directional_tilt(
    base_weights: pd.Series,
    tilt: dict[str, float],
    deviation_limit: float,
) -> pd.Series:
    assets = list(base_weights.index)
    weights = base_weights.reindex(assets).fillna(0.0).copy()
    raw_tilt = pd.Series({asset: tilt.get(asset, 0.0) for asset in assets}, dtype=float)
    raw_tilt = raw_tilt.clip(lower=-deviation_limit, upper=deviation_limit)
    if raw_tilt.sum() != 0:
        raw_tilt = raw_tilt - raw_tilt.sum() / len(raw_tilt)

    lower_bounds = (weights - deviation_limit).clip(lower=0.0)
    lower_bounds = lower_bounds.clip(lower=TAA_LOWER_BOUNDS.reindex(assets).fillna(0.0))
    upper_bounds = (weights + deviation_limit).clip(upper=1.0)
    upper_bounds = upper_bounds.clip(upper=TAA_UPPER_BOUNDS.reindex(assets).fillna(1.0))
    desired = weights + raw_tilt
    try:
        new_weights = project_weights_to_feasible(desired, assets, lower_bounds, upper_bounds)
    except RuntimeError:
        new_weights = project_weights_to_feasible(weights, assets, lower_bounds, upper_bounds)
    return new_weights.reindex(assets).fillna(0.0)


def _snap_to_index(date: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp | None:
    eligible = index[index <= date]
    if len(eligible) == 0:
        return None
    return pd.Timestamp(eligible[-1])


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


def signal_forecast_to_bl_taa_targets(
    signals: pd.Series,
    probabilities: pd.Series,
    forecasts: pd.DataFrame,
    saa_result: BacktestResult,
    returns: pd.DataFrame,
    *,
    decision_dates: list[pd.Timestamp] | None = None,
    deviation_limit: float = TAA_DEVIATION_LIMIT,
    covariance_window: int | None = BASE_WINDOW,
    risk_aversion: float = 4.0,
    bl_tau: float = 0.05,
    bl_confidence_floor: float = 0.35,
    bl_confidence_ceiling: float = 0.85,
    forecast_scale: float = 12.0,
    view_return_clip: float = 0.60,
    regime_tilt: dict[str, float] | None = None,
    regime_view_scale: float = 0.50,
) -> dict[pd.Timestamp, pd.Series]:
    """Convert XGB signals + neural return forecasts into BL/Markowitz targets."""
    if decision_dates is None:
        decision_dates = list(signals.dropna().index)

    targets: dict[pd.Timestamp, pd.Series] = {}
    for original_date in decision_dates:
        original_date = pd.Timestamp(original_date)
        snapped = _snap_to_index(original_date, returns.index)
        if snapped is None:
            continue

        signal_value = float(signals.loc[:original_date].iloc[-1]) if not signals.loc[:original_date].empty else 0.0
        base_w = base_weight_for_date(saa_result.weights_by_rebalance, snapped)
        if signal_value == 0:
            targets[snapped] = base_w.copy()
            continue

        active_assets = [asset for asset in returns.columns if returns.loc[:snapped, asset].notna().any()]
        history = history_slice(returns[active_assets].dropna(how="all"), snapped, covariance_window)
        if len(history) < 20:
            targets[snapped] = base_w.copy()
            continue

        covariance = history.cov(min_periods=max(20, min(len(history), 20))) * ANNUALIZATION
        covariance = covariance.loc[active_assets, active_assets].fillna(0.0)

        lower_bounds = (base_w.reindex(active_assets).fillna(0.0) - deviation_limit).clip(lower=0.0)
        lower_bounds = lower_bounds.clip(lower=TAA_LOWER_BOUNDS.reindex(active_assets).fillna(0.0))
        upper_bounds = (base_w.reindex(active_assets).fillna(0.0) + deviation_limit).clip(upper=1.0)
        upper_bounds = upper_bounds.clip(upper=TAA_UPPER_BOUNDS.reindex(active_assets).fillna(1.0))

        month_forecast = forecasts.reindex(columns=active_assets).loc[:original_date]
        if month_forecast.empty:
            targets[snapped] = base_w.copy()
            continue
        annual_views = month_forecast.iloc[-1].astype(float) * forecast_scale
        annual_views = annual_views.clip(lower=-view_return_clip, upper=view_return_clip)

        if regime_tilt:
            tilt_views = pd.Series(regime_tilt, dtype=float).reindex(active_assets).fillna(0.0)
            annual_views = annual_views.add(tilt_views * regime_view_scale, fill_value=0.0)

        probability = float(probabilities.loc[:original_date].iloc[-1]) if not probabilities.loc[:original_date].empty else 0.5
        confidence = bl_confidence_floor + (bl_confidence_ceiling - bl_confidence_floor) * probability
        confidence = float(np.clip(confidence, bl_confidence_floor, bl_confidence_ceiling))

        base_active = base_w.reindex(active_assets).fillna(0.0)
        if base_active.sum() <= 0:
            base_active = pd.Series(1.0 / len(active_assets), index=active_assets)
        else:
            base_active = base_active / base_active.sum()

        try:
            posterior = black_litterman_posterior(
                covariance=covariance,
                market_weights=base_active,
                views=annual_views,
                tau=bl_tau,
                confidence=confidence,
                risk_aversion=risk_aversion,
            )
            optimized = solve_mean_variance_with_bounds(
                covariance=covariance,
                expected_returns=posterior,
                active_assets=active_assets,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                risk_aversion=risk_aversion,
            )
        except RuntimeError:
            optimized = base_active.copy()

        weights = optimized.reindex(returns.columns).fillna(0.0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        targets[snapped] = weights

    return targets


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
        returns=returns,
        target_weights_by_date=combined,
        strategy_name=strategy_name,
    )


def run_bl_taa_backtest(
    signals: pd.Series,
    probabilities: pd.Series,
    forecasts: pd.DataFrame,
    saa_result: BacktestResult,
    returns: pd.DataFrame,
    *,
    decision_dates: list[pd.Timestamp] | None = None,
    deviation_limit: float = TAA_DEVIATION_LIMIT,
    covariance_window: int | None = BASE_WINDOW,
    risk_aversion: float = 4.0,
    bl_tau: float = 0.05,
    bl_confidence_floor: float = 0.35,
    bl_confidence_ceiling: float = 0.85,
    forecast_scale: float = 12.0,
    view_return_clip: float = 0.60,
    regime_tilt: dict[str, float] | None = None,
    regime_view_scale: float = 0.50,
    strategy_name: str = "SAA + XGB/Neural BL TAA",
) -> BacktestResult:
    if decision_dates is None:
        decision_dates = list(signals.dropna().index)

    signal_targets = signal_forecast_to_bl_taa_targets(
        signals=signals,
        probabilities=probabilities,
        forecasts=forecasts,
        saa_result=saa_result,
        returns=returns,
        decision_dates=decision_dates,
        deviation_limit=deviation_limit,
        covariance_window=covariance_window,
        risk_aversion=risk_aversion,
        bl_tau=bl_tau,
        bl_confidence_floor=bl_confidence_floor,
        bl_confidence_ceiling=bl_confidence_ceiling,
        forecast_scale=forecast_scale,
        view_return_clip=view_return_clip,
        regime_tilt=regime_tilt,
        regime_view_scale=regime_view_scale,
    )

    combined: dict[pd.Timestamp, pd.Series] = {}
    for date, w in saa_result.weights_by_rebalance.iterrows():
        snapped = _snap_to_index(pd.Timestamp(date), returns.index)
        if snapped is not None:
            combined[snapped] = w
    for date, w in signal_targets.items():
        combined[pd.Timestamp(date)] = w
    combined = dict(sorted(combined.items()))

    return run_strategy_backtest(
        returns=returns,
        target_weights_by_date=combined,
        strategy_name=strategy_name,
    )


__all__ = [
    "BacktestResult",
    "annual_rebalance_dates",
    "apply_taa_overlay",
    "base_weight_for_date",
    "black_litterman_posterior",
    "build_saa_targets",
    "compute_metrics",
    "feasible_for_assets",
    "feasible_returns",
    "first_feasible_date",
    "grouped_period_end_dates",
    "history_slice",
    "project_weights_to_feasible",
    "run_benchmark_backtest",
    "run_bl_taa_backtest",
    "run_saa_backtest",
    "run_strategy_backtest",
    "run_taa_backtest",
    "signal_forecast_to_bl_taa_targets",
    "signal_to_taa_targets",
    "solve_mean_variance_with_bounds",
    "solve_ips_min_variance",
    "solve_min_variance_with_bounds",
    "solve_target_return_frontier",
]
