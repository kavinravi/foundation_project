"""Demo of an autoresearch-style local search.

Holds the model architectures fixed at the baseline defaults, then sweeps
the decision threshold and (optionally) a "tilt magnitude scalar" applied
to the per-hypothesis ``meta.tilt`` vector.

This is **not** the full overnight autoresearch loop — it's a small,
deterministic grid you can run locally to sanity-check that the framework
finds IPS-improving configs and to seed the autoresearch agent's branch
with known-good starting points.

Usage::

    python -m backtesting.run_threshold_search
    python -m backtesting.run_threshold_search --hypotheses h1 h3 --models lstm xgb
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.core.backtest import (
    feasible_returns,
    run_benchmark_backtest,
    run_saa_backtest,
)
from backtesting.core.data import load_market_panel
from backtesting.core.ips import IPS_RETURN_TARGET, ips_pass_count
from backtesting.hypotheses import HYPOTHESIS_REGISTRY, build_hypothesis
from backtesting.models import MODEL_REGISTRY
from backtesting.models.base import ModelConfig
from backtesting.prepare import (
    ARTIFACTS_DIR,
    RunResult,
    append_results_tsv,
    run_taa_from_model,
    save_run_artifacts,
    score_run,
)
from backtesting.run_all_baselines import DEFAULT_HYPERPARAMETERS, DEFAULT_SEQ_LEN

ROOT = Path(__file__).resolve().parent
SEARCH_CSV = ARTIFACTS_DIR / "search_results.csv"

THRESHOLDS = [0.40, 0.50, 0.55, 0.60, 0.65]
TILT_SCALES = [1.0, 1.5]


def _model_factory(model_name: str, hp: dict, seq_len: int, seed: int):
    def _f():
        return MODEL_REGISTRY[model_name](
            config=ModelConfig(seed=seed, sequence_length=seq_len, extra=dict(hp))
        )

    return _f


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypotheses", nargs="*", default=list(HYPOTHESIS_REGISTRY.keys()))
    parser.add_argument("--models", nargs="*", default=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--thresholds", nargs="*", type=float, default=THRESHOLDS)
    parser.add_argument("--tilt-scales", nargs="*", type=float, default=TILT_SCALES)
    args = parser.parse_args()

    panel = load_market_panel()
    rows: list[dict] = []
    best_per_combo: dict[tuple[str, str], RunResult] = {}

    for hyp_name in args.hypotheses:
        base_hyp = build_hypothesis(hyp_name)
        base_tilt = dict(base_hyp.meta.tilt)
        for model_name in args.models:
            hp = DEFAULT_HYPERPARAMETERS[model_name]
            factory = _model_factory(model_name, hp, DEFAULT_SEQ_LEN, seed=42)
            for threshold in args.thresholds:
                for scale in args.tilt_scales:
                    hyp = build_hypothesis(hyp_name)
                    hyp.meta.tilt = {k: round(v * scale, 4) for k, v in base_tilt.items()}
                    label = f"{hyp_name}/{model_name}/thr={threshold}/scale={scale}"
                    print(f">>> {label}")
                    t0 = time.time()
                    try:
                        saa, taa, signal, folds, _ = run_taa_from_model(
                            hyp, factory, panel=panel, threshold=threshold
                        )
                    except Exception as exc:
                        print(f"   crash: {exc}")
                        continue
                    elapsed = time.time() - t0
                    res = score_run(
                        hypothesis=hyp,
                        model_name=model_name,
                        saa=saa,
                        taa=taa,
                        signal=signal,
                        folds=folds,
                        threshold=threshold,
                        sequence_length=DEFAULT_SEQ_LEN,
                        elapsed_seconds=elapsed,
                        notes=f"scale={scale}",
                    )
                    rows.append(
                        {
                            "hypothesis": hyp_name,
                            "model": model_name,
                            "threshold": threshold,
                            "tilt_scale": scale,
                            "ips_pass": res.ips_pass_count_taa,
                            "ips_loss": res.ips_loss_taa,
                            "return_pa": res.taa_metrics["total_return_pa"],
                            "vol_pa": res.taa_metrics["volatility_pa"],
                            "max_dd": res.taa_metrics["max_drawdown"],
                            "sharpe": res.taa_metrics["sharpe_rf2"],
                            "calmar": res.taa_metrics["calmar"],
                            "delta_dd": res.delta_drawdown,
                            "delta_ret": res.delta_return,
                            "n_signals_fired": res.n_signals_fired,
                            "elapsed_seconds": elapsed,
                        }
                    )
                    print(
                        f"   pass={res.ips_pass_count_taa}/3 loss={res.ips_loss_taa:.3f} "
                        f"DD={res.taa_metrics['max_drawdown']*100:.2f}% ret={res.taa_metrics['total_return_pa']*100:.2f}% "
                        f"signals={res.n_signals_fired} ({elapsed:.1f}s)"
                    )
                    append_results_tsv(
                        res,
                        status="search",
                        description=f"search thr={threshold} scale={scale}",
                    )
                    best = best_per_combo.get((hyp_name, model_name))
                    if best is None or res.ips_loss_taa < best.ips_loss_taa:
                        best_per_combo[(hyp_name, model_name)] = res
                        save_run_artifacts(res, signal, taa)

    df = pd.DataFrame(rows).sort_values("ips_loss")
    df.to_csv(SEARCH_CSV, index=False)
    print(f"\nSwept {len(rows)} configs. Saved to {SEARCH_CSV}")
    print("\nTop 10 configs by IPS loss:")
    print(df.head(10).to_string(index=False))

    print("\nBest per (hypothesis, model) combo:")
    summary = []
    for (hyp_name, model_name), res in best_per_combo.items():
        summary.append(
            {
                "hypothesis": hyp_name,
                "model": model_name,
                "ips_pass": res.ips_pass_count_taa,
                "ips_loss": res.ips_loss_taa,
                "return_pa": res.taa_metrics["total_return_pa"],
                "vol_pa": res.taa_metrics["volatility_pa"],
                "max_dd": res.taa_metrics["max_drawdown"],
                "sharpe": res.taa_metrics["sharpe_rf2"],
                "threshold": res.threshold,
            }
        )
    print(pd.DataFrame(summary).sort_values("ips_loss").to_string(index=False))

    (ARTIFACTS_DIR / "search_best_per_combo.json").write_text(
        json.dumps(
            {
                f"{k[0]}|{k[1]}": asdict(v)
                for k, v in best_per_combo.items()
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
