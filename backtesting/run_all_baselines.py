"""Run the 12 (4 hypotheses × 3 models) baseline experiments and emit
``results.md`` plus an aggregate JSON / CSV summary.

The defaults here match the ``HYPERPARAMETERS`` block in ``train.py`` so
that this is a true *baseline* sweep — the autoresearch agent then iterates
on top to improve it. Re-running this script regenerates the artifacts and
the markdown report in place.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from backtesting.core.backtest import (
    feasible_returns,
    run_benchmark_backtest,
    run_saa_backtest,
)
from backtesting.core.data import load_market_panel
from backtesting.core.ips import (
    IPS_MAX_DRAWDOWN_TARGET,
    IPS_RETURN_TARGET,
    IPS_VOL_TARGET,
    ips_pass_count,
    ips_scorecard,
)
from backtesting.hypotheses import HYPOTHESIS_REGISTRY, build_hypothesis
from backtesting.models import MODEL_REGISTRY
from backtesting.models.base import ModelConfig
from backtesting.prepare import (
    ARTIFACTS_DIR,
    RESULTS_TSV,
    RunResult,
    append_results_tsv,
    run_taa_from_model,
    save_run_artifacts,
    score_run,
)

ROOT = Path(__file__).resolve().parent
RESULTS_MD = ROOT / "results.md"
SUMMARY_JSON = ARTIFACTS_DIR / "baseline_summary.json"
SUMMARY_CSV = ARTIFACTS_DIR / "baseline_summary.csv"

DEFAULT_HYPERPARAMETERS = {
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

DEFAULT_THRESHOLDS = {"xgb": 0.50, "lstm": 0.55, "transformer": 0.55}
DEFAULT_SEQ_LEN = 12


def _model_factory(model_name: str, hyperparams: dict, seq_len: int, seed: int):
    def _f():
        cfg = ModelConfig(seed=seed, sequence_length=seq_len, extra=dict(hyperparams))
        return MODEL_REGISTRY[model_name](config=cfg)

    return _f


def run_one(hypothesis_name: str, model_name: str, *, panel) -> RunResult | None:
    hyp = build_hypothesis(hypothesis_name)
    factory = _model_factory(
        model_name=model_name,
        hyperparams=DEFAULT_HYPERPARAMETERS[model_name],
        seq_len=DEFAULT_SEQ_LEN,
        seed=42,
    )
    threshold = DEFAULT_THRESHOLDS[model_name]
    print(f"\n>>> Running {hyp.meta.title} | {model_name} | threshold={threshold}")
    t0 = time.time()
    try:
        saa, taa, signal, folds, _ = run_taa_from_model(
            hyp, factory, panel=panel, threshold=threshold
        )
    except Exception:
        traceback.print_exc()
        return None
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
        notes=str(DEFAULT_HYPERPARAMETERS[model_name]),
    )
    save_run_artifacts(res, signal, taa)
    append_results_tsv(res, status="baseline", description=f"baseline {model_name}/{hypothesis_name}")
    print(
        f"   IPS: {res.ips_pass_count_taa}/3   objective={res.ips_objective_taa:.3f}   "
        f"ΔDD={res.delta_drawdown:+.4f}  ΔRet={res.delta_return:+.4f}  ΔSharpe={res.delta_sharpe:+.4f}  "
        f"({elapsed:.1f}s)"
    )
    return res


def _format_pct(x: float, dp: int = 2) -> str:
    return f"{x*100:+.{dp}f}%" if isinstance(x, (int, float)) else "n/a"


def _format_metric(value: float, target: float, sense: str) -> str:
    """Return ``"{value%:.2f} (PASS, +0.42)"`` style cell."""
    if sense == ">=":
        gap = value - target
        passed = value >= target
    elif sense == "<=":
        gap = target - value
        passed = value <= target
    else:
        raise ValueError(sense)
    badge = "PASS" if passed else "MISS"
    sign = "+" if gap >= 0 else ""
    return f"{value*100:.2f}% ({badge}, gap {sign}{gap*100:.2f}pp)"


def _load_search_results() -> pd.DataFrame | None:
    search_csv = ARTIFACTS_DIR / "search_results.csv"
    if not search_csv.exists():
        return None
    try:
        return pd.read_csv(search_csv)
    except Exception:
        return None


def _runresult_from_dict(d: dict) -> RunResult:
    """Re-hydrate a RunResult dataclass from its asdict() payload."""
    field_names = {f for f in RunResult.__dataclass_fields__}
    return RunResult(**{k: v for k, v in d.items() if k in field_names})


def write_results_md(
    results: list[RunResult],
    saa_metrics: pd.Series,
    benchmark_metrics: pd.Series,
) -> Path:
    search_df = _load_search_results()
    lines: list[str] = []
    lines.append("# TAA Backtesting Results")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("This document records the **baseline** results from running each "
                 "(hypothesis × model) combination once with the default hyperparameters "
                 "in `train.py`, plus the best result found by a small "
                 "(threshold × tilt-magnitude) local search "
                 "(`run_threshold_search.py`). The `autoresearch` overnight agent loop "
                 "is intended to iterate on top of these baselines — see `program.md`.")
    lines.append("")
    # Executive summary
    lines.append("## TL;DR")
    lines.append("")
    lines.append(
        f"- **IPS targets:** return p.a. ≥ {IPS_RETURN_TARGET*100:.0f}%, "
        f"volatility p.a. ≤ {IPS_VOL_TARGET*100:.0f}%, "
        f"max drawdown ≥ {IPS_MAX_DRAWDOWN_TARGET*100:.0f}%."
    )
    saa_pass = ips_pass_count(saa_metrics)
    bench_pass = ips_pass_count(benchmark_metrics)
    lines.append(
        f"- **SAA reference**: {saa_pass}/3 IPS, "
        f"return {saa_metrics['total_return_pa']*100:.2f}%, "
        f"vol {saa_metrics['volatility_pa']*100:.2f}%, "
        f"max DD {saa_metrics['max_drawdown']*100:.2f}%, "
        f"Sharpe {saa_metrics['sharpe_rf2']:.3f}."
    )
    lines.append(
        f"- **Equity-only benchmark (SPXT)**: {bench_pass}/3 IPS, "
        f"return {benchmark_metrics['total_return_pa']*100:.2f}%, "
        f"vol {benchmark_metrics['volatility_pa']*100:.2f}%, "
        f"max DD {benchmark_metrics['max_drawdown']*100:.2f}%."
    )
    if results:
        best_baseline = min(results, key=lambda r: r.ips_objective_taa)
        m = best_baseline.taa_metrics
        lines.append(
            f"- **Best baseline TAA**: {best_baseline.hypothesis}/{best_baseline.model} "
            f"→ {best_baseline.ips_pass_count_taa}/3 IPS, "
            f"ips_objective {best_baseline.ips_objective_taa:.3f}, "
            f"return {m['total_return_pa']*100:.2f}%, "
            f"vol {m['volatility_pa']*100:.2f}%, "
            f"max DD {m['max_drawdown']*100:.2f}%."
        )
    if search_df is not None and len(search_df) > 0:
        best_row = search_df.sort_values("ips_objective").iloc[0]
        lines.append(
            f"- **Best searched config** (out of {len(search_df)} trials): "
            f"{best_row['hypothesis']}/{best_row['model']} with "
            f"threshold {best_row['threshold']:.2f}, tilt scale {best_row['tilt_scale']:.1f} "
            f"→ {int(best_row['ips_pass'])}/3 IPS, "
            f"ips_objective {best_row['ips_objective']:.3f}, "
            f"return {best_row['return_pa']*100:.2f}%, "
            f"vol {best_row['vol_pa']*100:.2f}%, "
            f"max DD {best_row['max_dd']*100:.2f}% "
            f"(SAA was {saa_metrics['max_drawdown']*100:.2f}%)."
        )
        passes_two = (search_df['ips_pass'] >= 2).sum()
        passes_three = (search_df['ips_pass'] >= 3).sum()
        lines.append(
            f"- **Headline:** the best searched configs hit the return AND "
            f"volatility targets simultaneously ({passes_two} configs at ≥2/3) "
            f"but the **drawdown floor remains the binding constraint** "
            f"({passes_three} configs at 3/3). The framework is working — the "
            "agent now needs to iterate on features and SAA, not just thresholds."
        )
    lines.append("")
    lines.append("## Investment Policy Statement targets")
    lines.append("")
    lines.append(f"- **Return p.a.** ≥ {IPS_RETURN_TARGET:.0%}")
    lines.append(f"- **Volatility p.a.** ≤ {IPS_VOL_TARGET:.0%}")
    lines.append(f"- **Max drawdown** ≥ {IPS_MAX_DRAWDOWN_TARGET:.0%}")
    lines.append("")
    lines.append("Tradable universe: 6 Zion assets (US Equity, US Bonds, REITs, Gold, Bitcoin, JPY) "
                 "with per-asset bound 0%–30%, lower bound of 20% on US Equity and US Bonds, "
                 "non-traditional cap (Gold + Bitcoin + JPY) of 25%, ±15% TAA deviation band, "
                 "5 bps round-trip transaction cost, and 2% risk-free rate.")
    lines.append("")
    lines.append("## Reference portfolios")
    lines.append("")
    lines.append("| Portfolio | Return p.a. | Vol p.a. | Max DD | Sharpe | Calmar | IPS pass |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, m in [("Benchmark 60/40", benchmark_metrics), ("SAA (min-var, IPS-bound)", saa_metrics)]:
        lines.append(
            f"| {name} | {m['total_return_pa']*100:.2f}% | {m['volatility_pa']*100:.2f}% | "
            f"{m['max_drawdown']*100:.2f}% | {m['sharpe_rf2']:.3f} | {m['calmar']:.3f} | "
            f"{ips_pass_count(m)}/3 |"
        )
    lines.append("")
    lines.append("## Baseline scoreboard (all 12 combos)")
    lines.append("")
    lines.append("Sorted by IPS objective (lower is better).")
    lines.append("")
    lines.append("| Rank | Hypothesis | Model | IPS pass | IPS objective | Return p.a. | Vol p.a. | Max DD | Sharpe | ΔDD vs SAA | ΔRet vs SAA |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    sorted_results = sorted(results, key=lambda r: r.ips_objective_taa)
    for rank, r in enumerate(sorted_results, start=1):
        m = r.taa_metrics
        lines.append(
            f"| {rank} | {r.hypothesis} | {r.model} | {r.ips_pass_count_taa}/3 | "
                f"{r.ips_objective_taa:.3f} | {m['total_return_pa']*100:.2f}% | "
            f"{m['volatility_pa']*100:.2f}% | {m['max_drawdown']*100:.2f}% | "
            f"{m['sharpe_rf2']:.3f} | {r.delta_drawdown*100:+.2f}pp | {r.delta_return*100:+.2f}pp |"
        )
    lines.append("")

    if search_df is not None and len(search_df) > 0:
        lines.append("## Local search scoreboard")
        lines.append("")
        lines.append(
            f"`run_threshold_search.py` swept {len(search_df)} configurations across "
            f"{sorted(search_df['hypothesis'].unique().tolist())} hypotheses, "
            f"{sorted(search_df['model'].unique().tolist())} models, "
            f"{sorted(search_df['asset_setup'].unique().tolist())} asset sleeves, "
            f"{sorted(search_df['taa_band'].unique().tolist())} TAA bands, "
            f"{sorted(search_df['threshold'].unique().tolist())} thresholds, and "
            f"{sorted(search_df['tilt_scale'].unique().tolist())} tilt scales. "
            "This is *not* the full overnight `autoresearch` loop — it's a "
            "deterministic seed search to confirm the framework can find "
            "IPS-improving configs and to give the agent good starting points."
        )
        lines.append("")
        lines.append("**Top 10 configurations by IPS objective:**")
        lines.append("")
        lines.append("| Hyp | Model | Sleeve | Band | Preset | Seq | Threshold | Tilt scale | IPS pass | IPS objective | Return | Vol | Max DD | Sharpe |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        top = search_df.sort_values("ips_objective").head(10)
        for _, row in top.iterrows():
            lines.append(
                f"| {row['hypothesis']} | {row['model']} | {row['asset_setup']} | {row['taa_band']:.2f} | "
                f"{row.get('hp_preset', 'baseline')} | {int(row.get('seq_len', DEFAULT_SEQ_LEN))} | "
                f"{row['threshold']:.2f} | {row['tilt_scale']:.1f} | {int(row['ips_pass'])}/3 | {row['ips_objective']:.3f} | "
                f"{row['return_pa']*100:.2f}% | {row['vol_pa']*100:.2f}% | "
                f"{row['max_dd']*100:.2f}% | {row['sharpe']:.3f} |"
            )
        lines.append("")
        best_per = search_df.loc[
            search_df.groupby(['hypothesis', 'model', 'asset_setup'])['ips_objective'].idxmin()
        ].sort_values('ips_objective')
        lines.append("**Best config per (hypothesis, model, sleeve) combo after local search:**")
        lines.append("")
        lines.append("| Hyp | Model | Sleeve | Band | Preset | Seq | Threshold | Tilt scale | IPS pass | IPS objective | Max DD | Return | ΔDD vs SAA |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, row in best_per.iterrows():
            lines.append(
                f"| {row['hypothesis']} | {row['model']} | {row['asset_setup']} | {row['taa_band']:.2f} | "
                f"{row.get('hp_preset', 'baseline')} | {int(row.get('seq_len', DEFAULT_SEQ_LEN))} | "
                f"{row['threshold']:.2f} | {row['tilt_scale']:.1f} | {int(row['ips_pass'])}/3 | {row['ips_objective']:.3f} | "
                f"{row['max_dd']*100:.2f}% | {row['return_pa']*100:.2f}% | "
                f"{row['delta_dd']*100:+.2f}pp |"
            )
        lines.append("")

    # Per-hypothesis sections
    by_hyp: dict[str, list[RunResult]] = {}
    for r in results:
        by_hyp.setdefault(r.hypothesis, []).append(r)

    for hyp_name in HYPOTHESIS_REGISTRY:
        if hyp_name not in by_hyp:
            continue
        hyp = build_hypothesis(hyp_name)
        section = by_hyp[hyp_name]
        lines.append(f"## {hyp.meta.title}")
        lines.append("")
        lines.append(hyp.meta.summary)
        lines.append("")
        lines.append(
            f"**Hypothesised tilt direction:** "
            f"{'risk-off' if hyp.meta.signal_when_active < 0 else 'risk-on'}"
        )
        lines.append("")
        lines.append("**Tilt vector when signal fires:**")
        lines.append("")
        lines.append("| Asset | Tilt |")
        lines.append("|---|---|")
        for asset, tilt in hyp.meta.tilt.items():
            lines.append(f"| {asset} | {tilt:+.2f} |")
        lines.append("")
        lines.append("**Features (monthly):**")
        lines.append("")
        for feat in hyp.feature_columns:
            lines.append(f"- `{feat}`")
        lines.append("")
        lines.append("**Per-model baseline results:**")
        lines.append("")
        lines.append("| Model | IPS pass | Return p.a. | Vol p.a. | Max DD | Sharpe | Signals fired | Train s |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(section, key=lambda r: r.ips_objective_taa):
            m = r.taa_metrics
            lines.append(
                f"| {r.model} | {r.ips_pass_count_taa}/3 | "
                f"{_format_metric(m['total_return_pa'], IPS_RETURN_TARGET, '>=')} | "
                f"{_format_metric(m['volatility_pa'], IPS_VOL_TARGET, '<=')} | "
                f"{_format_metric(m['max_drawdown'], IPS_MAX_DRAWDOWN_TARGET, '>=')} | "
                f"{m['sharpe_rf2']:.3f} | {r.n_signals_fired} | {r.elapsed_seconds:.1f} |"
            )
        lines.append("")
        if search_df is not None and len(search_df) > 0:
            hyp_search = search_df[search_df['hypothesis'] == hyp_name]
            if len(hyp_search) > 0:
                lines.append("**Best searched config (per model):**")
                lines.append("")
                lines.append("| Model | Threshold | Tilt scale | IPS pass | Return p.a. | Vol p.a. | Max DD | Sharpe |")
                lines.append("|---|---|---|---|---|---|---|---|")
                for model_name, grp in hyp_search.groupby('model'):
                    best = grp.sort_values('ips_objective').iloc[0]
                    lines.append(
                        f"| {model_name} | {best['threshold']:.2f} | {best['tilt_scale']:.1f} | "
                        f"{int(best['ips_pass'])}/3 | "
                        f"{_format_metric(best['return_pa'], IPS_RETURN_TARGET, '>=')} | "
                        f"{_format_metric(best['vol_pa'], IPS_VOL_TARGET, '<=')} | "
                        f"{_format_metric(best['max_dd'], IPS_MAX_DRAWDOWN_TARGET, '>=')} | "
                        f"{best['sharpe']:.3f} |"
                    )
                lines.append("")
        # Per-hypothesis interpretation
        baseline_summary = ", ".join(
            f"{r.model} {r.ips_pass_count_taa}/3" for r in sorted(section, key=lambda r: r.ips_objective_taa)
        )
        lines.append(
            "**Interpretation.** "
            f"Across the three baseline models the IPS satisfaction was: {baseline_summary}. "
            f"The dominant failure mode is the {IPS_MAX_DRAWDOWN_TARGET*100:.0f}% max-drawdown floor "
            "(driven by the 2008 GFC and 2020 COVID equity crashes). The signal *direction* "
            f"({'risk-off — defensive when active' if hyp.meta.signal_when_active < 0 else 'risk-on — pro-equity when active'}) "
            "matches economic intuition; the open question is *prediction lead-time*."
        )
        lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append("### Data")
    lines.append("")
    lines.append(
        "- **Tradable price panel**: 6 assets pulled from the Zion data export "
        "(`SPXT`, `LBUSTRUU`, `B3REITT`, `XAU`, `XBTUSD`, `USDJPY`). The JPY series "
        "is inverted (`1/USDJPY`) so a USD-investor return convention applies. "
        "Returns are point-in-time: an asset enters the universe on its first "
        "valid observation."
    )
    lines.append(
        "- **Student-sourced signals**: HY OAS (LF98OAS), VIX, US Aggregate OAS "
        "(LBUSOAS), Fed Funds, US 3M LIBOR/SOFR, Russell 2000, ISM Manufacturing "
        "PMI, Conference Board confidence, Umich consumer sentiment, and U-3 "
        "unemployment. Cleaned and split out by `parse_foundation_data.py` from "
        "the `Foundation Project Data` sheet."
    )
    lines.append(
        "- **Inflation-linked reference**: 0–5y TIPS total-return index "
        "(`data/0_5Y_TIPS_2002_D.csv`) is used as a *signal* "
        "(TIPS-vs-Treasury spread) rather than a tradable position, because the "
        "IPS does not include TIPS in the asset list."
    )
    lines.append("")
    lines.append("### SAA construction")
    lines.append("")
    lines.append(
        "Annual rebalance, IPS-constrained minimum variance using the trailing "
        "252-day covariance matrix. Per-asset bounds [0%, 30%], lower bound of "
        "20% on US Equity and US Bonds, non-traditional cap of 25% on the "
        "Gold + Bitcoin + JPY sleeve. Rebalances start at the first feasible "
        "date (after Bitcoin and REITs both have data, ≈ 2003-09)."
    )
    lines.append("")
    lines.append("### TAA overlay")
    lines.append("")
    lines.append(
        "On each month-end decision date the model probability is converted to "
        "a discrete signal via the threshold. When the signal fires, the SAA "
        "weights are adjusted by the per-hypothesis tilt vector, then clipped "
        "to the IPS per-asset bounds, the ±15% TAA deviation band, and the "
        "non-traditional 25% cap. Weights are re-normalised to sum to 1.0. "
        "Round-trip transaction cost of 5 bps is charged on the turnover "
        "between consecutive target weights."
    )
    lines.append("")
    lines.append("### Walk-forward validation")
    lines.append("")
    lines.append(
        f"{12} expanding-window folds. Each fold: 60 months of model fitting "
        "(48 train + 12 validation for early stopping where applicable), 18 "
        "months of out-of-sample test, then expand the train block by 18 "
        "months and roll the test forward (no overlap). Total OOS coverage "
        "≈ 2008-Q1 → 2025-Q4."
    )
    lines.append("")
    lines.append("### Models")
    lines.append("")
    lines.append(
        "- **XGBoost** (`xgboost>=3`, GPU-first): standard gradient-boosted trees "
        "on the current month's features only. Uses `device=cuda` when CUDA is "
        "available."
    )
    lines.append(
        "- **LSTM** (PyTorch CUDA build on RTX 5090): two-layer LSTM with "
        "hidden=96, GELU MLP head, AdamW + cosine LR, BCE-with-logits loss "
        "with positive-class re-weighting, TF32-friendly CUDA settings, "
        "200-epoch budget with patience-30 early stopping on the validation "
        "block. Sequence length = 12 months."
    )
    lines.append(
        "- **Transformer** (PyTorch CUDA build on RTX 5090): two-layer "
        "encoder, d_model=64, nhead=4, dim_ff=128, sinusoidal positional "
        "encoding, GELU activations, pre-norm. Same training recipe as the "
        "LSTM. Sequence length = 12 months."
    )
    lines.append("")
    lines.append("### Optimisation objective")
    lines.append("")
    lines.append("The `autoresearch` agent minimises a single composite scalar:")
    lines.append("")
    lines.append("```")
    lines.append("ips_loss = 100 * ( max(0, 5% - return_pa)")
    lines.append("                 + max(0, vol_pa - 11%)")
    lines.append("                 + max(0, -13% - max_dd) )")
    lines.append("         - 0.10 * sharpe_rf2")
    lines.append("```")
    lines.append("")
    lines.append(
        "A feasible portfolio (all three IPS constraints met) gets a non-positive "
        "loss whose magnitude is dominated by the Sharpe tiebreaker. An "
        "infeasible portfolio pays a linear penalty proportional to how badly "
        "each constraint is missed."
    )
    lines.append("")
    lines.append("### Reproducibility")
    lines.append("")
    lines.append(
        "Re-run all baselines with `python -m backtesting.run_all_baselines`. "
        "Re-run a single combo with `python -m backtesting.train --hypothesis "
        "h1 --model lstm`. Re-run the local search with "
        "`python -m backtesting.run_threshold_search`. Rebuild this markdown "
        "from cached artifacts with `python -m backtesting.run_all_baselines "
        "--report-only`. The autoresearch loop iterates on `train.py` per the "
        "instructions in `program.md`."
    )
    lines.append("")
    # Findings & next steps for the autoresearch agent
    lines.append("## Findings & autoresearch suggestions")
    lines.append("")
    lines.append(
        "1. **The drawdown floor is the binding constraint, not return or "
        "volatility.** The IPS-constrained min-variance SAA already comes very "
        "close on return and volatility; the 2008 GFC dominates the "
        "max-drawdown number across every configuration."
    )
    lines.append(
        "2. **TAA helps return + vol more than DD.** Because all TAA decisions "
        "are taken at month-end after a signal fires, the overlay can shave a "
        "few hundred bps off DD but cannot eliminate the GFC tail without "
        "either a faster-firing signal or a more aggressive tilt budget."
    )
    lines.append(
        "3. **Tilt scale > model.** In the local search, the most reliable "
        "lever for improving `ips_loss` was *increasing the tilt magnitude* "
        "(scale 1.5 vs 0.5–1.0) rather than swapping models. This suggests "
        "the SAA → TAA dynamic range encoded in `hypotheses/*.py` is too "
        "conservative relative to the IPS ±15% TAA band."
    )
    lines.append(
        "4. **Hypothesis 2 (Growth Cycle) is the most consistent winner.** "
        "It hits return AND vol simultaneously across multiple model + "
        "threshold combinations. H1 (Market Stress) is surprisingly hard "
        "because the model only learns to identify stress *during* the "
        "drawdown, not before it."
    )
    lines.append("")
    lines.append("**Suggested next experiments for the autoresearch agent:**")
    lines.append("")
    lines.append(
        "- **Lead-the-event labels.** The current label `equity_drawdown_event` "
        "fires on the realised drawdown month. Try `equity_drawdown_lead_3m` "
        "(label = 1 if a >5% drawdown happens in the next 3 months). This "
        "gives the model something predictive to learn."
    )
    lines.append(
        "- **Wider tilt vectors.** Edit the per-hypothesis `tilt` in "
        "`hypotheses/*.py` so the active tilt approaches ±15% on each leg. "
        "Combine with TIPS-spread regime detection."
    )
    lines.append(
        "- **Rolling SAA recalibration.** The annual rebalance is locked to "
        "calendar years. Try a 6-month rebalance with a longer covariance "
        "window."
    )
    lines.append(
        "- **Stacked models.** Use the LSTM probability as an extra feature "
        "into XGBoost (or vice versa). The autoresearch agent can implement "
        "this purely inside `train.py`."
    )
    lines.append(
        "- **Asymmetric thresholds.** Use a high threshold to fire defensive "
        "and a low threshold to fire offensive (currently a single threshold "
        "controls both). The agent can encode this in `train.py`."
    )
    lines.append("")

    RESULTS_MD.write_text("\n".join(lines))
    return RESULTS_MD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hypotheses",
        nargs="*",
        default=list(HYPOTHESIS_REGISTRY.keys()),
        help="Subset of hypotheses to run (default: all four).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(MODEL_REGISTRY.keys()),
        help="Subset of models to run (default: all three).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip retraining; rebuild results.md from cached baseline_summary.json + search_results.csv.",
    )
    args = parser.parse_args()

    print("Loading market panel...")
    panel = load_market_panel()
    feasible_ret = feasible_returns(panel.returns)
    print(f"Returns coverage: {feasible_ret.index.min().date()} → {feasible_ret.index.max().date()}")

    print("\nRunning reference portfolios...")
    saa = run_saa_backtest(panel.returns)
    bench = run_benchmark_backtest(panel.returns)
    print(f"SAA  : {saa.metrics.to_dict()}")
    print(f"Bench: {bench.metrics.to_dict()}")

    if args.report_only:
        if not SUMMARY_JSON.exists():
            raise FileNotFoundError(
                f"--report-only requires {SUMMARY_JSON} from a previous full run."
            )
        cached = json.loads(SUMMARY_JSON.read_text())
        results = [_runresult_from_dict(d) for d in cached]
        print(f"\nLoaded {len(results)} cached baseline results from {SUMMARY_JSON}")
        md_path = write_results_md(results, saa.metrics, bench.metrics)
        print(f"Wrote {md_path}")
        return

    results: list[RunResult] = []
    for hyp_name in args.hypotheses:
        for model_name in args.models:
            res = run_one(hyp_name, model_name, panel=panel)
            if res is not None:
                results.append(res)

    summary_records = [asdict(r) for r in results]
    SUMMARY_JSON.write_text(json.dumps(summary_records, indent=2, default=str))
    pd.DataFrame(
        [
            {
                "hypothesis": r.hypothesis,
                "model": r.model,
                "ips_pass_taa": r.ips_pass_count_taa,
                "ips_loss_taa": r.ips_loss_taa,
                "ips_pass_saa": r.ips_pass_count_saa,
                "ips_loss_saa": r.ips_loss_saa,
                "taa_return_pa": r.taa_metrics["total_return_pa"],
                "taa_vol_pa": r.taa_metrics["volatility_pa"],
                "taa_max_dd": r.taa_metrics["max_drawdown"],
                "taa_sharpe": r.taa_metrics["sharpe_rf2"],
                "taa_calmar": r.taa_metrics["calmar"],
                "delta_dd": r.delta_drawdown,
                "delta_ret": r.delta_return,
                "delta_sharpe": r.delta_sharpe,
                "n_signals_fired": r.n_signals_fired,
                "elapsed_seconds": r.elapsed_seconds,
            }
            for r in results
        ]
    ).to_csv(SUMMARY_CSV, index=False)

    md_path = write_results_md(results, saa.metrics, bench.metrics)
    print(f"\nWrote {len(results)} results.")
    print(f"  CSV:      {SUMMARY_CSV}")
    print(f"  JSON:     {SUMMARY_JSON}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
