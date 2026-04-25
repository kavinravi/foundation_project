"""Single-file experiment entry point — the autoresearch agent edits THIS file.

Usage::

    # Pick a (hypothesis, model) combo via env vars (or just edit the
    # constants below):
    HYPOTHESIS=h1 MODEL=lstm python backtesting/train.py

    # Or via CLI:
    python backtesting/train.py --hypothesis h1 --model transformer

The script:
1. Loads the market panel,
2. Builds the hypothesis-specific dataset,
3. Trains the chosen model in a walk-forward loop,
4. Runs the TAA backtest on the out-of-sample signals,
5. Prints a metric block ending in ``ips_loss: <value>`` (lower = better),
   plus per-IPS-constraint pass/miss numbers.

The autoresearch agent edits the ``HYPERPARAMETERS`` block + the
``HYPOTHESIS`` / ``MODEL`` constants and re-runs to iterate.

Things the agent can tune:
  - HYPERPARAMETERS dict (per-model knobs: depth, lr, hidden size, ...)
  - SIGNAL_THRESHOLD (decision boundary on the model probability)
  - SEQUENCE_LENGTH (look-back window for LSTM/Transformer)
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
MODEL = os.environ.get("MODEL", "xgb")           # xgb / lstm / transformer
SIGNAL_THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
SEQUENCE_LENGTH = int(os.environ.get("SEQ_LEN", "12"))
SEED = int(os.environ.get("SEED", "42"))

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
        "hidden": 96,
        "num_layers": 2,
        "dropout": 0.20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 200,
        "batch_size": 32,
        "patience": 30,
    },
    "transformer": {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.20,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "epochs": 200,
        "batch_size": 32,
        "patience": 30,
    },
}

# ============================================================================
# Below this line is the experiment harness. The agent typically does NOT
# need to edit any of this — it just sets the constants above.
# ============================================================================

from backtesting.hypotheses import build_hypothesis  # noqa: E402
from backtesting.models import build_model  # noqa: E402
from backtesting.models.base import ModelConfig  # noqa: E402
from backtesting.prepare import (  # noqa: E402
    append_results_tsv,
    run_taa_from_model,
    save_run_artifacts,
    score_run,
)


def make_model_factory(model_name: str, hyperparams: dict, seq_len: int, seed: int) -> Callable:
    def _factory():
        config = ModelConfig(seed=seed, sequence_length=seq_len, extra=dict(hyperparams))
        return build_model(model_name, config=config)

    return _factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis", default=HYPOTHESIS)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--threshold", type=float, default=SIGNAL_THRESHOLD)
    parser.add_argument("--seq-len", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--description", default="")
    parser.add_argument("--no-log", action="store_true", help="Skip writing to results.tsv")
    args = parser.parse_args()

    hp = dict(HYPERPARAMETERS[args.model])
    hyp = build_hypothesis(args.hypothesis)
    factory = make_model_factory(args.model, hp, args.seq_len, args.seed)

    print(f"=== {hyp.meta.title} | {args.model} | thr={args.threshold} | seq_len={args.seq_len} ===")
    print(f"Hypothesis: {hyp.meta.summary}")
    print(f"Hyperparameters: {hp}")

    t0 = time.time()
    saa, taa, signal, folds, fold_records = run_taa_from_model(
        hyp, factory, threshold=args.threshold
    )
    elapsed = time.time() - t0

    result = score_run(
        hypothesis=hyp,
        model_name=args.model,
        saa=saa,
        taa=taa,
        signal=signal,
        folds=folds,
        threshold=args.threshold,
        sequence_length=args.seq_len,
        elapsed_seconds=elapsed,
        notes=str(hp),
    )
    save_run_artifacts(result, signal, taa)

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
    print(f"delta_drawdown:        {result.delta_drawdown:+.4f}")
    print(f"delta_return:          {result.delta_return:+.4f}")
    print(f"delta_sharpe:          {result.delta_sharpe:+.4f}")
    print(f"elapsed_seconds:       {elapsed:.1f}")
    print(f"ips_loss:              {result.ips_loss_taa:.6f}")  # primary metric

    if not args.no_log:
        append_results_tsv(result, status="keep", description=args.description or f"{args.model}/{args.hypothesis}")


if __name__ == "__main__":
    main()
