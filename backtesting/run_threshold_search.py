"""Hybrid autoresearch seed sweep.

Runs a compact deterministic sweep across all four hypotheses using the
assignment-required stack:

- XGBoost for regime/signal classification
- LSTM or Transformer for next-month return forecasting
- Black-Litterman posterior expected returns
- Markowitz mean-variance TAA weights

The default sweep is intentionally small: 4 hypotheses x 2 hybrid models.
Broaden it with CLI flags once the smoke run is stable.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from backtesting.core.ips import ASSET_ORDER
from backtesting.hypotheses import HYPOTHESIS_REGISTRY, build_hypothesis
from backtesting.models import MODEL_REGISTRY
from backtesting.models.base import ModelConfig
from backtesting.models.return_forecaster import TorchReturnForecaster
from backtesting.prepare import (
    ARTIFACTS_DIR,
    RunResult,
    append_results_tsv,
    run_hybrid_bl_taa_from_models,
    save_run_artifacts,
    score_run,
)

ROOT = Path(__file__).resolve().parent
SEARCH_CSV = ARTIFACTS_DIR / "search_results.csv"

DEFAULT_XGB_HYPERPARAMETERS = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 250,
    "min_child_weight": 1,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
}

RETURN_HYPERPARAMETERS = {
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

ASSET_SETUPS = {
    "ips_available": list(ASSET_ORDER),
    "policy_core": ["US Equity", "US Treasuries", "US TIPS", "US REITs", "Gold"],
    "policy_plus_bitcoin": ["US Equity", "US Treasuries", "US TIPS", "US REITs", "Gold", "Bitcoin"],
}


def _xgb_factory(seed: int):
    def _factory():
        cfg = ModelConfig(seed=seed, extra=dict(DEFAULT_XGB_HYPERPARAMETERS))
        return MODEL_REGISTRY["xgb"](config=cfg)

    return _factory


def _return_factory(model_name: str, seq_len: int, seed: int):
    def _factory():
        cfg = ModelConfig(
            seed=seed,
            sequence_length=seq_len,
            extra=dict(RETURN_HYPERPARAMETERS[model_name]),
        )
        return TorchReturnForecaster(model_name, config=cfg)

    return _factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypotheses", nargs="*", default=list(HYPOTHESIS_REGISTRY.keys()))
    parser.add_argument("--return-models", nargs="*", default=["lstm", "transformer"])
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.55])
    parser.add_argument("--taa-bands", nargs="*", type=float, default=[0.50])
    parser.add_argument("--asset-setups", nargs="*", default=["ips_available"])
    parser.add_argument("--seq-lens", nargs="*", type=int, default=[6])
    parser.add_argument("--risk-aversions", nargs="*", type=float, default=[4.0])
    parser.add_argument("--bl-taus", nargs="*", type=float, default=[0.05])
    parser.add_argument("--regime-view-scales", nargs="*", type=float, default=[0.50])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows: list[dict] = []
    best_per_combo: dict[tuple[str, str], RunResult] = {}

    for hyp_name in args.hypotheses:
        for return_model in args.return_models:
            for asset_setup in args.asset_setups:
                active_assets = ASSET_SETUPS[asset_setup]
                for seq_len in args.seq_lens:
                    for threshold in args.thresholds:
                        for taa_band in args.taa_bands:
                            for risk_aversion in args.risk_aversions:
                                for bl_tau in args.bl_taus:
                                    for regime_view_scale in args.regime_view_scales:
                                        hyp = build_hypothesis(hyp_name)
                                        model_label = f"xgb_{return_model}"
                                        label = (
                                            f"{hyp_name}/{model_label}/assets={asset_setup}/seq={seq_len}/"
                                            f"thr={threshold}/band={taa_band}/risk={risk_aversion}/"
                                            f"tau={bl_tau}/regime_view={regime_view_scale}"
                                        )
                                        print(f">>> {label}")
                                        t0 = time.time()
                                        try:
                                            saa, taa, signal, proba, forecasts, folds, _ = run_hybrid_bl_taa_from_models(
                                                hyp,
                                                _xgb_factory(args.seed),
                                                _return_factory(return_model, seq_len, args.seed),
                                                threshold=threshold,
                                                active_assets=active_assets,
                                                deviation_limit=taa_band,
                                                risk_aversion=risk_aversion,
                                                bl_tau=bl_tau,
                                                regime_view_scale=regime_view_scale,
                                            )
                                        except Exception as exc:
                                            elapsed = time.time() - t0
                                            print(f"   crash after {elapsed:.1f}s: {exc}")
                                            rows.append(
                                                {
                                                    "hypothesis": hyp_name,
                                                    "model": model_label,
                                                    "asset_setup": asset_setup,
                                                    "seq_len": seq_len,
                                                    "threshold": threshold,
                                                    "taa_band": taa_band,
                                                    "risk_aversion": risk_aversion,
                                                    "bl_tau": bl_tau,
                                                    "regime_view_scale": regime_view_scale,
                                                    "status": "crash",
                                                    "error": str(exc),
                                                }
                                            )
                                            continue

                                        elapsed = time.time() - t0
                                        res = score_run(
                                            hypothesis=hyp,
                                            model_name=model_label,
                                            saa=saa,
                                            taa=taa,
                                            signal=signal,
                                            folds=folds,
                                            threshold=threshold,
                                            sequence_length=seq_len,
                                            elapsed_seconds=elapsed,
                                            notes=(
                                                f"asset_setup={asset_setup}; assets={active_assets}; "
                                                f"taa_band={taa_band}; risk_aversion={risk_aversion}; "
                                                f"bl_tau={bl_tau}; regime_view_scale={regime_view_scale}"
                                            ),
                                            active_assets=active_assets,
                                        )
                                        best_key = (hyp_name, model_label)
                                        prior_best = best_per_combo.get(best_key)
                                        status = (
                                            "keep"
                                            if prior_best is None
                                            or res.ips_objective_taa < prior_best.ips_objective_taa
                                            else "discard"
                                        )
                                        if status == "keep":
                                            best_per_combo[best_key] = res
                                            out = save_run_artifacts(res, signal, taa)
                                            proba.to_csv(out / "xgb_probability.csv", header=["probability"], index_label="date")
                                            forecasts.to_csv(out / "return_forecasts.csv", index_label="date")

                                        append_results_tsv(
                                            res,
                                            status="search" if status == "keep" else "discard",
                                            description=label,
                                        )
                                        row = {
                                            "hypothesis": hyp_name,
                                            "model": model_label,
                                            "asset_setup": asset_setup,
                                            "assets": ",".join(active_assets),
                                            "seq_len": seq_len,
                                            "threshold": threshold,
                                            "taa_band": taa_band,
                                            "risk_aversion": risk_aversion,
                                            "bl_tau": bl_tau,
                                            "regime_view_scale": regime_view_scale,
                                            "ips_pass": res.ips_pass_count_taa,
                                            "ips_objective": res.ips_objective_taa,
                                            "return_pa": res.taa_metrics["total_return_pa"],
                                            "vol_pa": res.taa_metrics["volatility_pa"],
                                            "max_dd": res.taa_metrics["max_drawdown"],
                                            "sharpe": res.taa_metrics["sharpe_rf2"],
                                            "calmar": res.taa_metrics["calmar"],
                                            "delta_dd": res.delta_drawdown,
                                            "delta_ret": res.delta_return,
                                            "n_signals_fired": res.n_signals_fired,
                                            "elapsed_seconds": elapsed,
                                            "status": status,
                                        }
                                        rows.append(row)
                                        print(
                                            f"   {status}: pass={res.ips_pass_count_taa}/3 "
                                            f"objective={res.ips_objective_taa:.3f} "
                                            f"ret={res.taa_metrics['total_return_pa']*100:.2f}% "
                                            f"vol={res.taa_metrics['volatility_pa']*100:.2f}% "
                                            f"DD={res.taa_metrics['max_drawdown']*100:.2f}% "
                                            f"signals={res.n_signals_fired} ({elapsed:.1f}s)"
                                        )

    df = pd.DataFrame(rows)
    if SEARCH_CSV.exists():
        try:
            existing = pd.read_csv(SEARCH_CSV)
            df = pd.concat([existing, df], ignore_index=True, sort=False)
        except Exception:
            pass
    if not df.empty and "ips_objective" in df.columns:
        df = df.sort_values("ips_objective", na_position="last")
    SEARCH_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SEARCH_CSV, index=False)
    print(f"\nSaved {len(df)} hybrid search rows to {SEARCH_CSV}")
    if not df.empty and "ips_objective" in df.columns:
        print("\nTop configs by IPS objective:")
        print(df.head(10).to_string(index=False))

    best_payload = {}
    if not df.empty and "ips_objective" in df.columns:
        valid = df.dropna(subset=["ips_objective"])
        if not valid.empty:
            best_rows = valid.loc[valid.groupby(["hypothesis", "model"])["ips_objective"].idxmin()]
            best_payload = {
                f"{row['hypothesis']}|{row['model']}": row.to_dict()
                for _, row in best_rows.iterrows()
            }
    if not best_payload:
        best_payload = {f"{k[0]}|{k[1]}": asdict(v) for k, v in best_per_combo.items()}
    (ARTIFACTS_DIR / "search_best_per_combo.json").write_text(
        json.dumps(best_payload, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
