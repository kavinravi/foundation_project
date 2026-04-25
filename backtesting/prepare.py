"""Read-only data preparation + evaluation harness for the autoresearch loop.

This file is the analogue of Karpathy's ``prepare.py`` — it contains the
fixed evaluation harness and helper utilities. **DO NOT MODIFY**. The
agent only ever edits ``train.py``.

What lives here:

* ``prepare_dataset`` — assembles the (X, y) frame for a hypothesis once.
* ``run_walk_forward`` — runs an arbitrary model factory through the walk-
  forward folds, generates monthly probabilities → signals, and runs a
  TAA backtest against the SAA baseline.
* ``score_run`` — converts a ``BacktestResult`` into a single ``ips_loss``
  scalar (the autoresearch metric, lower is better) plus the full IPS
  scorecard.
"""

from __future__ import annotations

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
    run_saa_backtest,
    run_taa_backtest,
)
from backtesting.core.data import MarketPanel, load_market_panel, monthly_resample
from backtesting.core.ips import (
    IPS_MAX_DRAWDOWN_TARGET,
    IPS_RETURN_TARGET,
    IPS_VOL_TARGET,
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
    threshold: float
    sequence_length: int
    folds_used: int
    n_signals_fired: int
    saa_metrics: dict[str, float]
    taa_metrics: dict[str, float]
    benchmark_metrics: dict[str, float] | None
    ips_pass_count_taa: int
    ips_pass_count_saa: int
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

    returns_for_test = feasible_returns(panel.returns)
    saa_result = run_saa_backtest(returns_for_test)

    taa_result = run_taa_backtest(
        signals=signal,
        saa_result=saa_result,
        returns=returns_for_test,
        decision_dates=list(signal.index),
        risk_off_tilt=hyp_obj.meta.tilt if signal_value_when_active < 0 else None,
        risk_on_tilt=hyp_obj.meta.tilt if signal_value_when_active > 0 else None,
        strategy_name=f"TAA[{hypothesis.meta.name}]",
    )
    return saa_result, taa_result, signal, folds, fold_records


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
) -> RunResult:
    saa_m = saa.metrics
    taa_m = taa.metrics
    bm_m = benchmark.metrics.to_dict() if benchmark is not None else None

    return RunResult(
        hypothesis=hypothesis.meta.name,
        model=model_name,
        threshold=threshold,
        sequence_length=sequence_length,
        folds_used=len(folds),
        n_signals_fired=int((signal != 0).sum()),
        saa_metrics={k: float(v) for k, v in saa_m.items()},
        taa_metrics={k: float(v) for k, v in taa_m.items()},
        benchmark_metrics={k: float(v) for k, v in bm_m.items()} if bm_m else None,
        ips_pass_count_taa=ips_pass_count(taa_m),
        ips_pass_count_saa=ips_pass_count(saa_m),
        ips_loss_taa=float(ips_loss(taa_m)),
        ips_loss_saa=float(ips_loss(saa_m)),
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
            "timestamp\thypothesis\tmodel\tthreshold\tips_loss\tips_pass\tdelta_dd\tdelta_ret\tstatus\tdescription\n"
        )
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{timestamp}\t{result.hypothesis}\t{result.model}\t{result.threshold:.3f}\t"
        f"{result.ips_loss_taa:.4f}\t{result.ips_pass_count_taa}\t"
        f"{result.delta_drawdown:.4f}\t{result.delta_return:.4f}\t{status}\t{description}\n"
    )
    with RESULTS_TSV.open("a") as f:
        f.write(line)


def save_run_artifacts(result: RunResult, signal: pd.Series, taa: BacktestResult) -> Path:
    """Persist a JSON summary + the signal series + monthly TAA returns."""
    tag = f"{result.hypothesis}_{result.model}"
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
    "run_taa_from_model",
    "run_walk_forward",
    "save_run_artifacts",
    "score_run",
]
