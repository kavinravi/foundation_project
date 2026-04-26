"""Single-file experiment entry point — the autoresearch agent edits THIS file.

Usage::

    # Pick a hypothesis and neural return forecaster via env vars:
    HYPOTHESIS=h1 RETURN_MODEL=lstm python backtesting/train.py

    # Or via CLI:
    python -m backtesting.train --hypothesis h1 --return-model transformer

The script:
1. Loads the market panel,
2. Builds the hypothesis-specific dataset,
3. Trains XGBoost as the regime/signal classifier in a walk-forward loop,
4. Trains an LSTM or Transformer to forecast next-month asset returns,
5. Blends neural return views with Black-Litterman and solves Markowitz weights,
6. Runs the TAA backtest on the out-of-sample signals and forecasts,
5. Prints a metric block ending in ``ips_objective: <value>`` (lower = better),
   plus per-IPS-constraint pass/miss numbers.

The autoresearch agent edits the ``HYPERPARAMETERS`` block + the
``HYPOTHESIS`` / ``RETURN_MODEL`` constants and re-runs to iterate.

Things the agent can tune:
  - ACTIVE_ASSETS (candidate tradable sleeve drawn from `data/`)
  - HYPERPARAMETERS dict (XGB classifier + neural return forecaster knobs)
  - SIGNAL_THRESHOLD (decision boundary on the model probability)
  - TAA_DEVIATION_LIMIT (how far the overlay may move from the SAA target)
  - SEQUENCE_LENGTH (look-back window for LSTM/Transformer)
  - BL / Markowitz parameters (risk aversion, tau, confidence, view scaling)
  - The hypothesis-specific feature list (edit the corresponding
    backtesting/hypotheses/h*.py file)

DO NOT modify backtesting/prepare.py — that's the locked evaluation harness.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Callable

# ============================================================================
# CONFIG (edit these, the agent + you)
# ============================================================================

HYPOTHESIS = os.environ.get("HYPOTHESIS", "h1")  # h1 / h2 / h3 / h4
RETURN_MODEL = os.environ.get("RETURN_MODEL", "lstm")  # lstm / transformer
SIGNAL_THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
SEQUENCE_LENGTH = int(os.environ.get("SEQ_LEN", "12"))
SEED = int(os.environ.get("SEED", "42"))
ACTIVE_ASSETS_RAW = os.environ.get("ASSETS", "")
TAA_DEVIATION_LIMIT = float(os.environ.get("TAA_BAND", "1.0"))
RISK_AVERSION = float(os.environ.get("RISK_AVERSION", "4.0"))
BL_TAU = float(os.environ.get("BL_TAU", "0.05"))
BL_CONFIDENCE_FLOOR = float(os.environ.get("BL_CONFIDENCE_FLOOR", "0.35"))
BL_CONFIDENCE_CEILING = float(os.environ.get("BL_CONFIDENCE_CEILING", "0.85"))
FORECAST_SCALE = float(os.environ.get("FORECAST_SCALE", "12.0"))
VIEW_RETURN_CLIP = float(os.environ.get("VIEW_RETURN_CLIP", "0.60"))
REGIME_VIEW_SCALE = float(os.environ.get("REGIME_VIEW_SCALE", "0.50"))

# Per-model hyperparameters. The autoresearch agent edits these
# (or adds new keys) to search for IPS-improving combinations.
HYPERPARAMETERS = {
    "xgb": {
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 250,
        "min_child_weight": 1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
    },
    "lstm": {
        "hidden": 64,
        "num_layers": 1,
        "dropout": 0.15,
        "lr": 8e-4,
        "weight_decay": 1e-4,
        "epochs": 80,
        "batch_size": 32,
        "patience": 12,
    },
    "transformer": {
        "d_model": 48,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 96,
        "dropout": 0.15,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "epochs": 80,
        "batch_size": 32,
        "patience": 12,
    },
}

# ============================================================================
# Below this line is the experiment harness. The agent typically does NOT
# need to edit any of this — it just sets the constants above.
# ============================================================================

from backtesting.hypotheses import build_hypothesis  # noqa: E402
from backtesting.core.ips import ASSET_ORDER  # noqa: E402
from backtesting.models import build_model  # noqa: E402
from backtesting.models.base import ModelConfig  # noqa: E402
from backtesting.models.return_forecaster import TorchReturnForecaster  # noqa: E402
from backtesting.prepare import (  # noqa: E402
    append_results_tsv,
    run_hybrid_bl_taa_from_models,
    save_run_artifacts,
    score_run,
)


def make_model_factory(model_name: str, hyperparams: dict, seq_len: int, seed: int) -> Callable:
    def _factory():
        config = ModelConfig(seed=seed, sequence_length=seq_len, extra=dict(hyperparams))
        return build_model(model_name, config=config)

    return _factory


def make_return_model_factory(model_name: str, hyperparams: dict, seq_len: int, seed: int) -> Callable:
    def _factory():
        config = ModelConfig(seed=seed, sequence_length=seq_len, extra=dict(hyperparams))
        return TorchReturnForecaster(model_name, config=config)

    return _factory


def parse_active_assets(raw: str | None, cli_assets: list[str] | None) -> list[str]:
    if cli_assets:
        return cli_assets
    if raw:
        return [asset.strip() for asset in raw.split(",") if asset.strip()]
    return list(ASSET_ORDER)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis", default=HYPOTHESIS)
    parser.add_argument(
        "--return-model",
        default=RETURN_MODEL,
        choices=["lstm", "transformer"],
        help="Neural return forecaster paired with the XGB signal classifier.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Deprecated alias. Use --return-model lstm|transformer.",
    )
    parser.add_argument("--threshold", type=float, default=SIGNAL_THRESHOLD)
    parser.add_argument("--seq-len", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--assets", nargs="*", default=None)
    parser.add_argument("--taa-band", type=float, default=TAA_DEVIATION_LIMIT)
    parser.add_argument("--risk-aversion", type=float, default=RISK_AVERSION)
    parser.add_argument("--bl-tau", type=float, default=BL_TAU)
    parser.add_argument("--bl-confidence-floor", type=float, default=BL_CONFIDENCE_FLOOR)
    parser.add_argument("--bl-confidence-ceiling", type=float, default=BL_CONFIDENCE_CEILING)
    parser.add_argument("--forecast-scale", type=float, default=FORECAST_SCALE)
    parser.add_argument("--view-return-clip", type=float, default=VIEW_RETURN_CLIP)
    parser.add_argument("--regime-view-scale", type=float, default=REGIME_VIEW_SCALE)
    parser.add_argument("--description", default="")
    parser.add_argument("--no-log", action="store_true", help="Skip writing to results.tsv")
    args = parser.parse_args()

    if args.model in {"lstm", "transformer"} and args.return_model == RETURN_MODEL:
        args.return_model = args.model
    elif args.model in {"xgb_lstm", "xgb-lstm"}:
        args.return_model = "lstm"
    elif args.model in {"xgb_transformer", "xgb-transformer"}:
        args.return_model = "transformer"

    xgb_hp = dict(HYPERPARAMETERS["xgb"])
    return_hp = dict(HYPERPARAMETERS[args.return_model])
    hyp = build_hypothesis(args.hypothesis)
    signal_factory = make_model_factory("xgb", xgb_hp, args.seq_len, args.seed)
    return_factory = make_return_model_factory(args.return_model, return_hp, args.seq_len, args.seed)
    active_assets = parse_active_assets(ACTIVE_ASSETS_RAW, args.assets)
    model_label = f"xgb_{args.return_model}"

    print(f"=== {hyp.meta.title} | {model_label} | thr={args.threshold} | seq_len={args.seq_len} ===")
    print(f"Hypothesis: {hyp.meta.summary}")
    print(f"Active assets: {active_assets}")
    print(f"TAA deviation limit: {args.taa_band}")
    print(f"XGB signal hyperparameters: {xgb_hp}")
    print(f"{args.return_model} return hyperparameters: {return_hp}")
    print(
        "BL/Markowitz: "
        f"risk_aversion={args.risk_aversion}, tau={args.bl_tau}, "
        f"confidence=[{args.bl_confidence_floor}, {args.bl_confidence_ceiling}], "
        f"forecast_scale={args.forecast_scale}, view_clip={args.view_return_clip}, "
        f"regime_view_scale={args.regime_view_scale}"
    )

    t0 = time.time()
    saa, taa, signal, proba, forecasts, folds, fold_records = run_hybrid_bl_taa_from_models(
        hyp,
        signal_factory,
        return_factory,
        threshold=args.threshold,
        active_assets=active_assets,
        deviation_limit=args.taa_band,
        risk_aversion=args.risk_aversion,
        bl_tau=args.bl_tau,
        bl_confidence_floor=args.bl_confidence_floor,
        bl_confidence_ceiling=args.bl_confidence_ceiling,
        forecast_scale=args.forecast_scale,
        view_return_clip=args.view_return_clip,
        regime_view_scale=args.regime_view_scale,
    )
    elapsed = time.time() - t0

    result = score_run(
        hypothesis=hyp,
        model_name=model_label,
        saa=saa,
        taa=taa,
        signal=signal,
        folds=folds,
        threshold=args.threshold,
        sequence_length=args.seq_len,
        elapsed_seconds=elapsed,
        notes=(
            f"assets={active_assets}; taa_band={args.taa_band}; xgb_hp={xgb_hp}; "
            f"return_model={args.return_model}; return_hp={return_hp}; "
            f"risk_aversion={args.risk_aversion}; bl_tau={args.bl_tau}; "
            f"forecast_scale={args.forecast_scale}; regime_view_scale={args.regime_view_scale}"
        ),
        active_assets=active_assets,
    )
    artifact_dir = save_run_artifacts(result, signal, taa)
    proba.to_csv(artifact_dir / "xgb_probability.csv", header=["probability"], index_label="date")
    forecasts.to_csv(artifact_dir / "return_forecasts.csv", index_label="date")

    print()
    print("---")
    for fr in fold_records:
        print(
            f"fold {fr.get('fold')}: train {fr.get('train_start')} .. {fr.get('train_end')} "
            f"-> test {fr.get('test_start')} .. {fr.get('test_end')} "
            f"| n_train={fr.get('n_train')} pos_rate={fr.get('positive_rate_train')}"
        )
    print("---")
    print(f"signals_fired:         {result.n_signals_fired}/{len(signal)}")
    print(f"saa_return_pa:         {saa.metrics['total_return_pa']:.4f}")
    print(f"taa_return_pa:         {taa.metrics['total_return_pa']:.4f}")
    print(f"saa_vol_pa:            {saa.metrics['volatility_pa']:.4f}")
    print(f"taa_vol_pa:            {taa.metrics['volatility_pa']:.4f}")
    print(f"saa_max_drawdown:      {saa.metrics['max_drawdown']:.4f}")
    print(f"taa_max_drawdown:      {taa.metrics['max_drawdown']:.4f}")
    print(f"saa_sharpe:            {saa.metrics['sharpe_rf2']:.4f}")
    print(f"taa_sharpe:            {taa.metrics['sharpe_rf2']:.4f}")
    print(f"saa_calmar:            {saa.metrics['calmar']:.4f}")
    print(f"taa_calmar:            {taa.metrics['calmar']:.4f}")
    print(f"ips_pass_taa:          {result.ips_pass_count_taa}/3")
    print(f"ips_pass_saa:          {result.ips_pass_count_saa}/3")
    print(f"return_gap_taa:        {result.return_gap_taa:.4f}")
    print(f"vol_gap_taa:           {result.vol_gap_taa:.4f}")
    print(f"drawdown_gap_taa:      {result.drawdown_gap_taa:.4f}")
    print(f"delta_drawdown:        {result.delta_drawdown:+.4f}")
    print(f"delta_return:          {result.delta_return:+.4f}")
    print(f"delta_sharpe:          {result.delta_sharpe:+.4f}")
    print(f"elapsed_seconds:       {elapsed:.1f}")
    print(f"ips_objective:         {result.ips_objective_taa:.6f}")  # primary metric

    if not args.no_log:
        append_results_tsv(
            result,
            status="keep",
            description=args.description or f"{model_label}/{args.hypothesis}/band={args.taa_band}",
        )


if __name__ == "__main__":
    main()
