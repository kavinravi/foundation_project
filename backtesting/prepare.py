"""Read-only data preparation + evaluation harness for the autoresearch loop.

This file is the analogue of Karpathy's ``prepare.py`` — it contains the
fixed evaluation harness and helper utilities. **DO NOT MODIFY**. The
agent only ever edits ``train.py``.

What lives here:

* ``prepare_dataset`` — assembles the (X, y) frame for a hypothesis once.
* ``run_walk_forward`` — runs an arbitrary model factory through the walk-
  forward folds, generates monthly probabilities → signals, and runs a
  TAA backtest against the SAA baseline.
* ``score_run`` — converts a ``BacktestResult`` into a single constraint-first
  IPS objective derived only from annualized return, annualized volatility,
  and maximum drawdown.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from backtesting.core.backtest import (
    BacktestResult,
    feasible_returns,
    run_bl_taa_backtest,
    run_saa_backtest,
    run_taa_backtest,
)
from backtesting.core.data import MarketPanel, load_market_panel, monthly_resample
from backtesting.core.ips import (
    IPS_MAX_DRAWDOWN_TARGET,
    IPS_RETURN_TARGET,
    IPS_VOL_TARGET,
    ips_constraint_gaps,
    ips_objective,
    ips_loss,
    ips_pass_count,
    ips_scorecard,
)
from backtesting.core.walk_forward import Fold, make_walk_forward_folds
from backtesting.hypotheses import build_hypothesis
from backtesting.hypotheses.base import Hypothesis
from backtesting.models import build_model
from backtesting.models.base import BaseModel, ModelConfig

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
RESULTS_TSV = ROOT / "results.tsv"

# Fixed walk-forward setup so experiments are comparable across runs.
# Total span = initial_train + n_folds * step. With H1's first feature month
# at ~2003-01, this covers 60 + 12*18 = 276 months ≈ 2003-01 → 2026-01,
# putting the first test fold over the 2008 GFC and the last over 2024-25.
FIXED_FOLDS = dict(
    initial_train_months=60,
    val_months=12,
    test_months=18,
    n_folds=12,
    step_months=18,
)


@dataclass
class RunResult:
    hypothesis: str
    model: str
    active_assets: list[str]
    threshold: float
    sequence_length: int
    folds_used: int
    n_signals_fired: int
    saa_metrics: dict[str, float]
    taa_metrics: dict[str, float]
    benchmark_metrics: dict[str, float] | None
    ips_pass_count_taa: int
    ips_pass_count_saa: int
    ips_objective_taa: float
    ips_objective_saa: float
    return_gap_taa: float
    vol_gap_taa: float
    drawdown_gap_taa: float
    ips_loss_taa: float
    ips_loss_saa: float
    delta_drawdown: float
    delta_return: float
    delta_vol: float
    delta_sharpe: float
    elapsed_seconds: float
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def prepare_dataset(hypothesis_name: str, panel: MarketPanel | None = None) -> tuple[Hypothesis, pd.DataFrame, pd.Series]:
    panel = panel or load_market_panel()
    hyp = build_hypothesis(hypothesis_name)
    X, y = hyp.assemble_dataset(panel)
    return hyp, X, y


def make_folds(monthly_index: pd.DatetimeIndex) -> list[Fold]:
    return make_walk_forward_folds(monthly_index, **FIXED_FOLDS)


def run_walk_forward(
    hypothesis: Hypothesis,
    model_factory: Callable[[], BaseModel],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    threshold: float = 0.5,
    folds: list[Fold] | None = None,
) -> tuple[pd.Series, list[Fold], list[dict]]:
    """Train ``model_factory`` over each fold's train/val block and emit
    out-of-sample monthly probabilities concatenated across the test blocks."""

    folds = folds or make_folds(pd.DatetimeIndex(X.index))
    if not folds:
        raise RuntimeError("No walk-forward folds could be constructed (insufficient history).")

    proba_segments: list[pd.Series] = []
    fold_records: list[dict] = []
    for fold in folds:
        X_train = X.loc[X.index.isin(fold.train)]
        y_train = y.loc[y.index.isin(fold.train)]
        X_val = X.loc[X.index.isin(fold.validation)]
        y_val = y.loc[y.index.isin(fold.validation)]
        X_test = X.loc[X.index.isin(fold.test)]

        if X_train.empty or y_train.empty or X_test.empty:
            fold_records.append({"fold": fold.fold_id, "skipped": True})
            continue

        model = model_factory()
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        probas = model.predict_proba(X_test)
        if not probas.empty:
            proba_segments.append(probas)
        fold_records.append(
            {
                "fold": fold.fold_id,
                "train_start": fold.train.min(),
                "train_end": fold.train.max(),
                "test_start": fold.test.min(),
                "test_end": fold.test.max(),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "positive_rate_train": float(y_train.mean()),
                "test_positive_rate_pred": float((probas >= threshold).mean()) if not probas.empty else float("nan"),
            }
        )

    if not proba_segments:
        return pd.Series(dtype=float), folds, fold_records
    full_proba = pd.concat(proba_segments).sort_index()
    full_proba = full_proba[~full_proba.index.duplicated(keep="last")]
    return full_proba, folds, fold_records


def run_taa_from_model(
    hypothesis: Hypothesis,
    model_factory: Callable[[], BaseModel],
    *,
    panel: MarketPanel | None = None,
    threshold: float = 0.5,
    active_assets: list[str] | None = None,
    deviation_limit: float | None = None,
) -> tuple[BacktestResult, BacktestResult, pd.Series, list[Fold], list[dict]]:
    """Full pipeline: fold the data, train the model, and run the TAA backtest.

    Returns ``(saa_result, taa_result, signals, folds, fold_records)``.
    """
    panel = panel or load_market_panel()
    panel = panel.slice(end=pd.Timestamp("2026-04-15"))
    hyp_obj, X, y = prepare_dataset(hypothesis.meta.name, panel=panel)

    proba, folds, fold_records = run_walk_forward(
        hyp_obj, model_factory, X, y, threshold=threshold
    )
    if proba.empty:
        raise RuntimeError("Model produced no out-of-sample probabilities.")

    signal_value_when_active = hyp_obj.meta.signal_when_active
    signal = pd.Series(
        np.where(proba >= threshold, signal_value_when_active, 0),
        index=proba.index,
        dtype=int,
    )

    returns_full = feasible_returns(panel.returns, asset_universe=active_assets)
    returns_for_test = _trim_returns_to_oos_decision_window(returns_full, signal.index)
    saa_result = run_saa_backtest(
        returns_full,
        asset_universe=active_assets,
        run_start=returns_for_test.index.min(),
    )

    taa_result = run_taa_backtest(
        signals=signal,
        saa_result=saa_result,
        returns=returns_for_test,
        decision_dates=list(signal.index),
        deviation_limit=deviation_limit if deviation_limit is not None else 1.0,
        risk_off_tilt=hyp_obj.meta.tilt if signal_value_when_active < 0 else None,
        risk_on_tilt=hyp_obj.meta.tilt if signal_value_when_active > 0 else None,
        strategy_name=f"TAA[{hypothesis.meta.name}]",
    )
    return saa_result, taa_result, signal, folds, fold_records


def _trim_returns_to_oos_decision_window(
    returns: pd.DataFrame,
    decision_index: pd.Index,
) -> pd.DataFrame:
    """Start returns at the first OOS decision date snapped to trading days."""
    if len(decision_index) == 0:
        return returns
    first_decision = pd.Timestamp(pd.DatetimeIndex(decision_index).min())
    eligible = returns.index[returns.index <= first_decision]
    if len(eligible) > 0:
        start = eligible[-1]
    else:
        future = returns.index[returns.index >= first_decision]
        if len(future) == 0:
            raise RuntimeError("No return observations overlap the OOS decision window.")
        start = future[0]
    return returns.loc[start:]


def _compound_monthly_returns(returns: pd.DataFrame) -> pd.DataFrame:
    def _compound(group: pd.DataFrame) -> pd.Series:
        valid = group.notna().any(axis=0)
        compounded = (1.0 + group.fillna(0.0)).prod(axis=0) - 1.0
        compounded.loc[~valid] = np.nan
        return compounded

    return returns.resample("ME").apply(_compound)


def run_hybrid_bl_taa_from_models(
    hypothesis: Hypothesis,
    signal_model_factory: Callable[[], BaseModel],
    return_model_factory: Callable[[], object],
    *,
    panel: MarketPanel | None = None,
    threshold: float = 0.5,
    active_assets: list[str] | None = None,
    deviation_limit: float | None = None,
    risk_aversion: float = 4.0,
    bl_tau: float = 0.05,
    bl_confidence_floor: float = 0.35,
    bl_confidence_ceiling: float = 0.85,
    forecast_scale: float = 12.0,
    view_return_clip: float = 0.60,
    regime_view_scale: float = 0.50,
) -> tuple[BacktestResult, BacktestResult, pd.Series, pd.Series, pd.DataFrame, list[Fold], list[dict]]:
    """Hybrid TAA pipeline: XGB signal + neural returns + BL Markowitz weights."""
    panel = panel or load_market_panel()
    panel = panel.slice(end=pd.Timestamp("2026-04-15"))
    hyp_obj, X, y = prepare_dataset(hypothesis.meta.name, panel=panel)
    returns_full = feasible_returns(panel.returns, asset_universe=active_assets)
    active_cols = list(returns_full.columns)
    monthly_forward_returns = _compound_monthly_returns(returns_full[active_cols]).shift(-1)
    monthly_forward_returns = monthly_forward_returns.reindex(X.index)

    folds = make_folds(pd.DatetimeIndex(X.index))
    if not folds:
        raise RuntimeError("No walk-forward folds could be constructed (insufficient history).")

    proba_segments: list[pd.Series] = []
    forecast_segments: list[pd.DataFrame] = []
    fold_records: list[dict] = []

    for fold in folds:
        X_train = X.loc[X.index.isin(fold.train)]
        y_train = y.loc[y.index.isin(fold.train)]
        X_val = X.loc[X.index.isin(fold.validation)]
        y_val = y.loc[y.index.isin(fold.validation)]
        X_test = X.loc[X.index.isin(fold.test)]

        if X_train.empty or y_train.empty or X_test.empty:
            fold_records.append({"fold": fold.fold_id, "skipped": True})
            continue

        signal_model = signal_model_factory()
        signal_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        probas = signal_model.predict_proba(X_test)
        if not probas.empty:
            proba_segments.append(probas)

        return_model = return_model_factory()
        y_ret_train = monthly_forward_returns.reindex(X_train.index)
        y_ret_val = monthly_forward_returns.reindex(X_val.index) if not X_val.empty else None
        return_model.fit(X_train, y_ret_train, X_val=X_val, y_val=y_ret_val)
        context = X.loc[X.index <= X_test.index.max()]
        fold_forecasts = return_model.predict(context).reindex(X_test.index)
        forecast_segments.append(fold_forecasts)

        fold_records.append(
            {
                "fold": fold.fold_id,
                "train_start": fold.train.min(),
                "train_end": fold.train.max(),
                "test_start": fold.test.min(),
                "test_end": fold.test.max(),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "positive_rate_train": float(y_train.mean()),
                "test_positive_rate_pred": float((probas >= threshold).mean()) if not probas.empty else float("nan"),
            }
        )

    if not proba_segments or not forecast_segments:
        raise RuntimeError("Hybrid model produced no out-of-sample probabilities or forecasts.")

    full_proba = pd.concat(proba_segments).sort_index()
    full_proba = full_proba[~full_proba.index.duplicated(keep="last")]
    full_forecasts = pd.concat(forecast_segments).sort_index()
    full_forecasts = full_forecasts[~full_forecasts.index.duplicated(keep="last")]
    full_forecasts = full_forecasts.reindex(full_proba.index).ffill().fillna(0.0)

    signal_value_when_active = hyp_obj.meta.signal_when_active
    signal = pd.Series(
        np.where(full_proba >= threshold, signal_value_when_active, 0),
        index=full_proba.index,
        dtype=int,
    )

    returns_for_test = _trim_returns_to_oos_decision_window(returns_full, signal.index)
    saa_result = run_saa_backtest(
        returns_full,
        asset_universe=active_cols,
        run_start=returns_for_test.index.min(),
    )
    taa_result = run_bl_taa_backtest(
        signals=signal,
        probabilities=full_proba,
        forecasts=full_forecasts,
        saa_result=saa_result,
        returns=returns_for_test,
        decision_dates=list(signal.index),
        deviation_limit=deviation_limit if deviation_limit is not None else 1.0,
        risk_aversion=risk_aversion,
        bl_tau=bl_tau,
        bl_confidence_floor=bl_confidence_floor,
        bl_confidence_ceiling=bl_confidence_ceiling,
        forecast_scale=forecast_scale,
        view_return_clip=view_return_clip,
        regime_tilt=hyp_obj.meta.tilt,
        regime_view_scale=regime_view_scale,
        strategy_name=f"HybridBL[{hypothesis.meta.name}]",
    )
    return saa_result, taa_result, signal, full_proba, full_forecasts, folds, fold_records


def score_run(
    hypothesis: Hypothesis,
    model_name: str,
    saa: BacktestResult,
    taa: BacktestResult,
    signal: pd.Series,
    folds: list[Fold],
    *,
    threshold: float,
    sequence_length: int,
    elapsed_seconds: float,
    benchmark: BacktestResult | None = None,
    notes: str = "",
    active_assets: list[str] | None = None,
) -> RunResult:
    saa_m = saa.metrics
    taa_m = taa.metrics
    bm_m = benchmark.metrics.to_dict() if benchmark is not None else None
    ret_gap, vol_gap, dd_gap = ips_constraint_gaps(taa_m)
    taa_objective = float(ips_objective(taa_m))
    saa_objective = float(ips_objective(saa_m))

    return RunResult(
        hypothesis=hypothesis.meta.name,
        model=model_name,
        active_assets=list(active_assets or list(taa.weights_by_rebalance.columns)),
        threshold=threshold,
        sequence_length=sequence_length,
        folds_used=len(folds),
        n_signals_fired=int((signal != 0).sum()),
        saa_metrics={k: float(v) for k, v in saa_m.items()},
        taa_metrics={k: float(v) for k, v in taa_m.items()},
        benchmark_metrics={k: float(v) for k, v in bm_m.items()} if bm_m else None,
        ips_pass_count_taa=ips_pass_count(taa_m),
        ips_pass_count_saa=ips_pass_count(saa_m),
        ips_objective_taa=taa_objective,
        ips_objective_saa=saa_objective,
        return_gap_taa=float(ret_gap),
        vol_gap_taa=float(vol_gap),
        drawdown_gap_taa=float(dd_gap),
        ips_loss_taa=taa_objective,  # legacy alias for older scripts/artifacts
        ips_loss_saa=saa_objective,
        delta_drawdown=float(taa_m["max_drawdown"] - saa_m["max_drawdown"]),
        delta_return=float(taa_m["total_return_pa"] - saa_m["total_return_pa"]),
        delta_vol=float(taa_m["volatility_pa"] - saa_m["volatility_pa"]),
        delta_sharpe=float(taa_m["sharpe_rf2"] - saa_m["sharpe_rf2"]),
        elapsed_seconds=elapsed_seconds,
        notes=notes,
    )


def append_results_tsv(result: RunResult, status: str = "keep", description: str = "") -> None:
    """Autoresearch-style append. One row per experiment."""
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(
            "timestamp\thypothesis\tmodel\tthreshold\tips_objective\tips_pass\tdelta_dd\tdelta_ret\tstatus\tdescription\n"
        )
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    assets_label = ",".join(result.active_assets)
    full_description = f"assets={assets_label} | {description}" if description else f"assets={assets_label}"
    line = (
        f"{timestamp}\t{result.hypothesis}\t{result.model}\t{result.threshold:.3f}\t"
        f"{result.ips_objective_taa:.4f}\t{result.ips_pass_count_taa}\t"
        f"{result.delta_drawdown:.4f}\t{result.delta_return:.4f}\t{status}\t{full_description}\n"
    )
    with RESULTS_TSV.open("a") as f:
        f.write(line)


def save_run_artifacts(result: RunResult, signal: pd.Series, taa: BacktestResult) -> Path:
    """Persist a JSON summary + the signal series + monthly TAA returns."""
    asset_sig = hashlib.md5(",".join(result.active_assets).encode("utf-8")).hexdigest()[:8]
    tag = f"{result.hypothesis}_{result.model}_{asset_sig}"
    out = ARTIFACTS_DIR / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result.to_dict(), indent=2, default=str))
    signal.to_csv(out / "signal.csv", header=["signal"], index_label="date")
    taa.daily_returns.to_csv(out / "taa_daily_returns.csv", header=["return"], index_label="date")
    taa.weights_by_rebalance.to_csv(out / "taa_weights.csv", index_label="rebalance_date")
    return out


__all__ = [
    "ARTIFACTS_DIR",
    "FIXED_FOLDS",
    "RESULTS_TSV",
    "RunResult",
    "append_results_tsv",
    "make_folds",
    "prepare_dataset",
    "run_hybrid_bl_taa_from_models",
    "run_taa_from_model",
    "run_walk_forward",
    "save_run_artifacts",
    "score_run",
]
