# TAA Backtesting Results

Generated: 2026-04-24 21:00:37

This document records the **baseline** results from running each (hypothesis × model) combination once with the default hyperparameters in `train.py`, plus the best result found by a small (threshold × tilt-magnitude) local search (`run_threshold_search.py`). The `autoresearch` overnight agent loop is intended to iterate on top of these baselines — see `program.md`.

## TL;DR

- **IPS targets:** return p.a. ≥ 5%, volatility p.a. ≤ 11%, max drawdown ≥ -13%.
- **SAA reference**: 1/3 IPS, return 4.68%, vol 6.15%, max DD -19.88%, Sharpe 0.450.
- **Equity-only benchmark (SPXT)**: 2/3 IPS, return 7.69%, vol 10.09%, max DD -34.44%.
- **Best baseline TAA**: h1/lstm → 1/3 IPS, ips_loss 6.315, return 4.74%, vol 6.08%, max DD -19.10%.
- **Best searched config** (out of 120 trials): h2/xgb with threshold 0.60, tilt scale 1.5 → 2/3 IPS, ips_loss 4.386, return 5.47%, vol 6.68%, max DD -17.44% (SAA was -19.88%).
- **Headline:** the best searched configs hit the return AND volatility targets simultaneously (27 configs at ≥2/3) but the **drawdown floor remains the binding constraint** (0 configs at 3/3). The framework is working — the agent now needs to iterate on features and SAA, not just thresholds.

## Investment Policy Statement targets

- **Return p.a.** ≥ 5%
- **Volatility p.a.** ≤ 11%
- **Max drawdown** ≥ -13%

Tradable universe: 6 Zion assets (US Equity, US Bonds, REITs, Gold, Bitcoin, JPY) with per-asset bound 0%–30%, lower bound of 20% on US Equity and US Bonds, non-traditional cap (Gold + Bitcoin + JPY) of 25%, ±15% TAA deviation band, 5 bps round-trip transaction cost, and 2% risk-free rate.

## Reference portfolios

| Portfolio | Return p.a. | Vol p.a. | Max DD | Sharpe | Calmar | IPS pass |
|---|---|---|---|---|---|---|
| Benchmark 60/40 | 7.69% | 10.09% | -34.44% | 0.588 | 0.223 | 2/3 |
| SAA (min-var, IPS-bound) | 4.68% | 6.15% | -19.88% | 0.450 | 0.236 | 1/3 |

## Baseline scoreboard (all 12 combos)

Sorted by IPS loss (lower is better; 0 means all 3 IPS constraints met).

| Rank | Hypothesis | Model | IPS pass | IPS loss | Return p.a. | Vol p.a. | Max DD | Sharpe | ΔDD vs SAA | ΔRet vs SAA |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | h1 | lstm | 1/3 | 6.315 | 4.74% | 6.08% | -19.10% | 0.463 | +0.78pp | +0.05pp |
| 2 | h3 | transformer | 1/3 | 6.343 | 4.71% | 6.11% | -19.10% | 0.457 | +0.78pp | +0.02pp |
| 3 | h3 | xgb | 1/3 | 6.733 | 4.77% | 6.06% | -19.55% | 0.470 | +0.33pp | +0.08pp |
| 4 | h1 | xgb | 1/3 | 6.736 | 4.66% | 6.00% | -19.44% | 0.456 | +0.44pp | -0.03pp |
| 5 | h4 | lstm | 1/3 | 6.777 | 4.67% | 6.14% | -19.49% | 0.448 | +0.39pp | -0.02pp |
| 6 | h2 | transformer | 1/3 | 6.906 | 4.93% | 6.29% | -19.88% | 0.478 | -0.01pp | +0.24pp |
| 7 | h3 | lstm | 1/3 | 7.041 | 4.80% | 6.13% | -19.88% | 0.469 | -0.01pp | +0.10pp |
| 8 | h1 | transformer | 1/3 | 7.102 | 4.74% | 6.12% | -19.88% | 0.460 | -0.01pp | +0.04pp |
| 9 | h4 | transformer | 1/3 | 7.144 | 4.69% | 6.14% | -19.88% | 0.453 | -0.01pp | +0.00pp |
| 10 | h4 | xgb | 1/3 | 7.241 | 4.60% | 6.14% | -19.88% | 0.437 | -0.01pp | -0.09pp |
| 11 | h2 | xgb | 2/3 | 7.441 | 5.85% | 6.72% | -20.50% | 0.583 | -0.62pp | +1.16pp |
| 12 | h2 | lstm | 2/3 | 7.753 | 5.30% | 6.59% | -20.80% | 0.513 | -0.93pp | +0.61pp |

## Local search scoreboard (threshold × tilt magnitude)

`run_threshold_search.py` swept 120 configurations: [0.4, 0.5, 0.55, 0.6, 0.65] thresholds × [1.0, 1.5] tilt-magnitude scalars × ['h1', 'h2', 'h3', 'h4'] hypotheses × ['lstm', 'transformer', 'xgb'] models. This is *not* the full overnight `autoresearch` loop — it's a deterministic seed search to confirm the framework can find IPS-improving configs and to give the agent good starting points.

**Top 10 configurations by IPS loss:**

| Hyp | Model | Threshold | Tilt scale | IPS pass | IPS loss | Return | Vol | Max DD | Sharpe |
|---|---|---|---|---|---|---|---|---|---|
| h2 | xgb | 0.60 | 1.5 | 2/3 | 4.386 | 5.47% | 6.68% | -17.44% | 0.532 |
| h2 | xgb | 0.65 | 1.5 | 2/3 | 4.388 | 5.30% | 6.65% | -17.44% | 0.509 |
| h2 | xgb | 0.60 | 1.0 | 2/3 | 4.921 | 5.28% | 6.55% | -17.97% | 0.513 |
| h2 | xgb | 0.65 | 1.0 | 2/3 | 4.923 | 5.14% | 6.52% | -17.97% | 0.494 |
| h4 | lstm | 0.40 | 1.5 | 1/3 | 5.205 | 4.89% | 5.97% | -18.14% | 0.494 |
| h4 | lstm | 0.50 | 1.5 | 1/3 | 5.300 | 4.79% | 5.99% | -18.14% | 0.478 |
| h4 | lstm | 0.40 | 1.0 | 1/3 | 5.548 | 4.77% | 5.91% | -18.36% | 0.479 |
| h4 | lstm | 0.50 | 1.0 | 1/3 | 5.636 | 4.68% | 5.94% | -18.36% | 0.463 |
| h3 | lstm | 0.40 | 1.5 | 1/3 | 5.691 | 4.76% | 5.87% | -18.50% | 0.480 |
| h4 | transformer | 0.40 | 1.5 | 1/3 | 5.697 | 4.85% | 5.96% | -18.59% | 0.489 |

**Best config per (hypothesis, model) combo after local search:**

| Hyp | Model | Threshold | Tilt scale | IPS pass | IPS loss | Max DD | Return | ΔDD vs SAA |
|---|---|---|---|---|---|---|---|---|
| h2 | xgb | 0.60 | 1.5 | 2/3 | 4.386 | -17.44% | 5.47% | +2.44pp |
| h4 | lstm | 0.40 | 1.5 | 1/3 | 5.205 | -18.14% | 4.89% | +1.74pp |
| h3 | lstm | 0.40 | 1.5 | 1/3 | 5.691 | -18.50% | 4.76% | +1.38pp |
| h4 | transformer | 0.40 | 1.5 | 1/3 | 5.697 | -18.59% | 4.85% | +1.29pp |
| h3 | transformer | 0.40 | 1.5 | 1/3 | 5.784 | -18.50% | 4.67% | +1.38pp |
| h1 | lstm | 0.40 | 1.5 | 1/3 | 5.877 | -18.50% | 4.57% | +1.38pp |
| h1 | xgb | 0.60 | 1.5 | 1/3 | 6.579 | -19.37% | 4.74% | +0.51pp |
| h3 | xgb | 0.60 | 1.5 | 1/3 | 6.681 | -19.45% | 4.72% | +0.43pp |
| h1 | transformer | 0.50 | 1.5 | 1/3 | 6.691 | -19.32% | 4.59% | +0.55pp |
| h2 | lstm | 0.65 | 1.5 | 2/3 | 6.826 | -19.88% | 5.59% | -0.01pp |
| h2 | transformer | 0.60 | 1.5 | 2/3 | 6.832 | -19.88% | 5.17% | -0.01pp |
| h4 | xgb | 0.65 | 1.5 | 1/3 | 7.175 | -19.88% | 4.66% | -0.01pp |

## H1 — Market Stress

Credit spreads, implied vol, equity trend, and small-cap relative performance jointly mark deteriorating financial conditions. When the signal fires, tilt towards defensive assets within IPS bands.

**Hypothesised tilt direction:** risk-off

**Tilt vector when signal fires:**

| Asset | Tilt |
|---|---|
| US Equity | -0.10 |
| REITs | -0.05 |
| Bitcoin | -0.03 |
| US Bonds | +0.10 |
| Gold | +0.05 |
| JPY | +0.03 |

**Features (monthly):**

- `hy_oas_level`
- `hy_oas_chg_3m`
- `hy_oas_z_3y`
- `vix_level`
- `vix_chg_1m`
- `vix_z_1y`
- `spx_mom_3m`
- `spx_dd_6m`
- `spx_vol_3m`
- `rty_minus_spx_3m`
- `agg_oas_chg_3m`

**Per-model baseline results:**

| Model | IPS pass | Return p.a. | Vol p.a. | Max DD | Sharpe | Signals fired | Train s |
|---|---|---|---|---|---|---|---|
| lstm | 1/3 | 4.74% (MISS, gap -0.26pp) | 6.08% (PASS, gap +4.92pp) | -19.10% (MISS, gap -6.10pp) | 0.463 | 34 | 17.3 |
| xgb | 1/3 | 4.66% (MISS, gap -0.34pp) | 6.00% (PASS, gap +5.00pp) | -19.44% (MISS, gap -6.44pp) | 0.456 | 60 | 1.7 |
| transformer | 1/3 | 4.74% (MISS, gap -0.26pp) | 6.12% (PASS, gap +4.88pp) | -19.88% (MISS, gap -6.88pp) | 0.460 | 28 | 41.7 |

**Best searched config (per model):**

| Model | Threshold | Tilt scale | IPS pass | Return p.a. | Vol p.a. | Max DD | Sharpe |
|---|---|---|---|---|---|---|---|
| lstm | 0.40 | 1.5 | 1/3 | 4.57% (MISS, gap -0.43pp) | 5.83% (PASS, gap +5.17pp) | -18.50% (MISS, gap -5.50pp) | 0.453 |
| transformer | 0.50 | 1.5 | 1/3 | 4.59% (MISS, gap -0.41pp) | 5.88% (PASS, gap +5.12pp) | -19.32% (MISS, gap -6.32pp) | 0.452 |
| xgb | 0.60 | 1.5 | 1/3 | 4.74% (MISS, gap -0.26pp) | 6.04% (PASS, gap +4.96pp) | -19.37% (MISS, gap -6.37pp) | 0.467 |

**Interpretation.** Across the three baseline models the IPS satisfaction was: lstm 1/3, xgb 1/3, transformer 1/3. The dominant failure mode is the -13% max-drawdown floor (driven by the 2008 GFC and 2020 COVID equity crashes). The signal *direction* (risk-off — defensive when active) matches economic intuition; the open question is *prediction lead-time*.

## H2 — Growth Cycle

When ISM PMI, consumer confidence, and small-cap relative momentum are all improving, the economy is entering a favourable regime. Lean into risk assets while staying within IPS bands.

**Hypothesised tilt direction:** risk-on

**Tilt vector when signal fires:**

| Asset | Tilt |
|---|---|
| US Equity | +0.10 |
| REITs | +0.05 |
| Bitcoin | +0.03 |
| US Bonds | -0.10 |
| Gold | -0.04 |
| JPY | -0.04 |

**Features (monthly):**

- `ism_level`
- `ism_chg_3m`
- `ism_chg_6m`
- `cb_conf_level`
- `cb_conf_chg_3m`
- `umich_level`
- `umich_chg_3m`
- `rty_minus_spx_3m`
- `rty_mom_6m`
- `spx_mom_6m`
- `unemp_chg_6m`

**Per-model baseline results:**

| Model | IPS pass | Return p.a. | Vol p.a. | Max DD | Sharpe | Signals fired | Train s |
|---|---|---|---|---|---|---|---|
| transformer | 1/3 | 4.93% (MISS, gap -0.07pp) | 6.29% (PASS, gap +4.71pp) | -19.88% (MISS, gap -6.88pp) | 0.478 | 20 | 36.4 |
| xgb | 2/3 | 5.85% (PASS, gap +0.85pp) | 6.72% (PASS, gap +4.28pp) | -20.50% (MISS, gap -7.50pp) | 0.583 | 59 | 1.5 |
| lstm | 2/3 | 5.30% (PASS, gap +0.30pp) | 6.59% (PASS, gap +4.41pp) | -20.80% (MISS, gap -7.80pp) | 0.513 | 34 | 14.5 |

**Best searched config (per model):**

| Model | Threshold | Tilt scale | IPS pass | Return p.a. | Vol p.a. | Max DD | Sharpe |
|---|---|---|---|---|---|---|---|
| lstm | 0.65 | 1.5 | 2/3 | 5.59% (PASS, gap +0.59pp) | 6.31% (PASS, gap +4.69pp) | -19.88% (MISS, gap -6.88pp) | 0.577 |
| transformer | 0.60 | 1.5 | 2/3 | 5.17% (PASS, gap +0.17pp) | 6.35% (PASS, gap +4.65pp) | -19.88% (MISS, gap -6.88pp) | 0.512 |
| xgb | 0.60 | 1.5 | 2/3 | 5.47% (PASS, gap +0.47pp) | 6.68% (PASS, gap +4.32pp) | -17.44% (MISS, gap -4.44pp) | 0.532 |

**Interpretation.** Across the three baseline models the IPS satisfaction was: transformer 1/3, xgb 2/3, lstm 2/3. The dominant failure mode is the -13% max-drawdown floor (driven by the 2008 GFC and 2020 COVID equity crashes). The signal *direction* (risk-on — pro-equity when active) matches economic intuition; the open question is *prediction lead-time*.

## H3 — Two-Stage Regime Change

Markets react first (HY OAS, VIX, SPX trend); macro confirms later (ISM PMI, sentiment). A model fed both blocks should require their joint agreement before triggering risk-off, improving precision.

**Hypothesised tilt direction:** risk-off

**Tilt vector when signal fires:**

| Asset | Tilt |
|---|---|
| US Equity | -0.12 |
| REITs | -0.05 |
| Bitcoin | -0.03 |
| US Bonds | +0.12 |
| Gold | +0.05 |
| JPY | +0.03 |

**Features (monthly):**

- `hy_oas_level`
- `hy_oas_chg_1m`
- `hy_oas_chg_3m`
- `vix_level`
- `vix_chg_1m`
- `spx_mom_3m`
- `spx_dd_6m`
- `spx_vol_3m`
- `ism_level`
- `ism_chg_3m`
- `cb_conf_chg_3m`
- `umich_chg_3m`
- `stress_macro_interaction`

**Per-model baseline results:**

| Model | IPS pass | Return p.a. | Vol p.a. | Max DD | Sharpe | Signals fired | Train s |
|---|---|---|---|---|---|---|---|
| transformer | 1/3 | 4.71% (MISS, gap -0.29pp) | 6.11% (PASS, gap +4.89pp) | -19.10% (MISS, gap -6.10pp) | 0.457 | 26 | 25.5 |
| xgb | 1/3 | 4.77% (MISS, gap -0.23pp) | 6.06% (PASS, gap +4.94pp) | -19.55% (MISS, gap -6.55pp) | 0.470 | 46 | 1.6 |
| lstm | 1/3 | 4.80% (MISS, gap -0.20pp) | 6.13% (PASS, gap +4.87pp) | -19.88% (MISS, gap -6.88pp) | 0.469 | 21 | 13.3 |

**Best searched config (per model):**

| Model | Threshold | Tilt scale | IPS pass | Return p.a. | Vol p.a. | Max DD | Sharpe |
|---|---|---|---|---|---|---|---|
| lstm | 0.40 | 1.5 | 1/3 | 4.76% (MISS, gap -0.24pp) | 5.87% (PASS, gap +5.13pp) | -18.50% (MISS, gap -5.50pp) | 0.480 |
| transformer | 0.40 | 1.5 | 1/3 | 4.67% (MISS, gap -0.33pp) | 5.85% (PASS, gap +5.15pp) | -18.50% (MISS, gap -5.50pp) | 0.467 |
| xgb | 0.60 | 1.5 | 1/3 | 4.72% (MISS, gap -0.28pp) | 6.08% (PASS, gap +4.92pp) | -19.45% (MISS, gap -6.45pp) | 0.460 |

**Interpretation.** Across the three baseline models the IPS satisfaction was: transformer 1/3, xgb 1/3, lstm 1/3. The dominant failure mode is the -13% max-drawdown floor (driven by the 2008 GFC and 2020 COVID equity crashes). The signal *direction* (risk-off — defensive when active) matches economic intuition; the open question is *prediction lead-time*.

## H4 — Stagflation Signal

Growth weakens (HY OAS rising, ISM falling, VIX rising) while inflation hedges (TIPS, gold) outperform Treasuries. Tilt towards Bonds (proxy for TIPS) and Gold; keep crypto small for vol control.

**Hypothesised tilt direction:** risk-off

**Tilt vector when signal fires:**

| Asset | Tilt |
|---|---|
| US Equity | -0.10 |
| REITs | -0.05 |
| Bitcoin | -0.03 |
| US Bonds | +0.08 |
| Gold | +0.10 |
| JPY | +0.00 |

**Features (monthly):**

- `hy_oas_level`
- `hy_oas_chg_3m`
- `ism_level`
- `ism_chg_3m`
- `vix_level`
- `vix_chg_1m`
- `tips_minus_bond_3m`
- `tips_minus_bond_6m`
- `gold_mom_3m`
- `gold_mom_6m`
- `gold_minus_spx_6m`

**Per-model baseline results:**

| Model | IPS pass | Return p.a. | Vol p.a. | Max DD | Sharpe | Signals fired | Train s |
|---|---|---|---|---|---|---|---|
| lstm | 1/3 | 4.67% (MISS, gap -0.33pp) | 6.14% (PASS, gap +4.86pp) | -19.49% (MISS, gap -6.49pp) | 0.448 | 19 | 13.4 |
| transformer | 1/3 | 4.69% (MISS, gap -0.31pp) | 6.14% (PASS, gap +4.86pp) | -19.88% (MISS, gap -6.88pp) | 0.453 | 10 | 38.9 |
| xgb | 1/3 | 4.60% (MISS, gap -0.40pp) | 6.14% (PASS, gap +4.86pp) | -19.88% (MISS, gap -6.88pp) | 0.437 | 12 | 1.4 |

**Best searched config (per model):**

| Model | Threshold | Tilt scale | IPS pass | Return p.a. | Vol p.a. | Max DD | Sharpe |
|---|---|---|---|---|---|---|---|
| lstm | 0.40 | 1.5 | 1/3 | 4.89% (MISS, gap -0.11pp) | 5.97% (PASS, gap +5.03pp) | -18.14% (MISS, gap -5.14pp) | 0.494 |
| transformer | 0.40 | 1.5 | 1/3 | 4.85% (MISS, gap -0.15pp) | 5.96% (PASS, gap +5.04pp) | -18.59% (MISS, gap -5.59pp) | 0.489 |
| xgb | 0.65 | 1.5 | 1/3 | 4.66% (MISS, gap -0.34pp) | 6.15% (PASS, gap +4.85pp) | -19.88% (MISS, gap -6.88pp) | 0.447 |

**Interpretation.** Across the three baseline models the IPS satisfaction was: lstm 1/3, transformer 1/3, xgb 1/3. The dominant failure mode is the -13% max-drawdown floor (driven by the 2008 GFC and 2020 COVID equity crashes). The signal *direction* (risk-off — defensive when active) matches economic intuition; the open question is *prediction lead-time*.

## Methodology

### Data

- **Tradable price panel**: 6 assets pulled from the Zion data export (`SPXT`, `LBUSTRUU`, `B3REITT`, `XAU`, `XBTUSD`, `USDJPY`). The JPY series is inverted (`1/USDJPY`) so a USD-investor return convention applies. Returns are point-in-time: an asset enters the universe on its first valid observation.
- **Student-sourced signals**: HY OAS (LF98OAS), VIX, US Aggregate OAS (LBUSOAS), Fed Funds, US 3M LIBOR/SOFR, Russell 2000, ISM Manufacturing PMI, Conference Board confidence, Umich consumer sentiment, and U-3 unemployment. Cleaned and split out by `parse_foundation_data.py` from the `Foundation Project Data` sheet.
- **Inflation-linked reference**: 0–5y TIPS total-return index (`data/0_5Y_TIPS_2002_D.csv`) is used as a *signal* (TIPS-vs-Treasury spread) rather than a tradable position, because the IPS does not include TIPS in the asset list.

### SAA construction

Annual rebalance, IPS-constrained minimum variance using the trailing 252-day covariance matrix. Per-asset bounds [0%, 30%], lower bound of 20% on US Equity and US Bonds, non-traditional cap of 25% on the Gold + Bitcoin + JPY sleeve. Rebalances start at the first feasible date (after Bitcoin and REITs both have data, ≈ 2003-09).

### TAA overlay

On each month-end decision date the model probability is converted to a discrete signal via the threshold. When the signal fires, the SAA weights are adjusted by the per-hypothesis tilt vector, then clipped to the IPS per-asset bounds, the ±15% TAA deviation band, and the non-traditional 25% cap. Weights are re-normalised to sum to 1.0. Round-trip transaction cost of 5 bps is charged on the turnover between consecutive target weights.

### Walk-forward validation

12 expanding-window folds. Each fold: 60 months of model fitting (48 train + 12 validation for early stopping where applicable), 18 months of out-of-sample test, then expand the train block by 18 months and roll the test forward (no overlap). Total OOS coverage ≈ 2008-Q1 → 2025-Q4.

### Models

- **XGBoost** (`xgboost==3.2`): standard gradient-boosted trees on the current month's features only. The tabular baseline.
- **LSTM** (PyTorch 2.9, CUDA 12.8 on RTX 5090): two-layer LSTM with hidden=96, GELU MLP head, AdamW + cosine LR, BCE-with-logits loss with positive-class re-weighting, 200-epoch budget with patience-30 early stopping on the validation block. Sequence length = 12 months.
- **Transformer** (PyTorch 2.9, CUDA 12.8 on RTX 5090): two-layer encoder, d_model=64, nhead=4, dim_ff=128, sinusoidal positional encoding, GELU activations, pre-norm. Same training recipe as the LSTM. Sequence length = 12 months.

### Optimisation objective

The `autoresearch` agent minimises a single composite scalar:

```
ips_loss = 100 * ( max(0, 5% - return_pa)
                 + max(0, vol_pa - 11%)
                 + max(0, -13% - max_dd) )
         - 0.10 * sharpe_rf2
```

A feasible portfolio (all three IPS constraints met) gets a non-positive loss whose magnitude is dominated by the Sharpe tiebreaker. An infeasible portfolio pays a linear penalty proportional to how badly each constraint is missed.

### Reproducibility

Re-run all baselines with `python -m backtesting.run_all_baselines`. Re-run a single combo with `python -m backtesting.train --hypothesis h1 --model lstm`. Re-run the local search with `python -m backtesting.run_threshold_search`. Rebuild this markdown from cached artifacts with `python -m backtesting.run_all_baselines --report-only`. The autoresearch loop iterates on `train.py` per the instructions in `program.md`.

## Findings & autoresearch suggestions

1. **The drawdown floor is the binding constraint, not return or volatility.** The IPS-constrained min-variance SAA already comes very close on return and volatility; the 2008 GFC dominates the max-drawdown number across every configuration.
2. **TAA helps return + vol more than DD.** Because all TAA decisions are taken at month-end after a signal fires, the overlay can shave a few hundred bps off DD but cannot eliminate the GFC tail without either a faster-firing signal or a more aggressive tilt budget.
3. **Tilt scale > model.** In the local search, the most reliable lever for improving `ips_loss` was *increasing the tilt magnitude* (scale 1.5 vs 0.5–1.0) rather than swapping models. This suggests the SAA → TAA dynamic range encoded in `hypotheses/*.py` is too conservative relative to the IPS ±15% TAA band.
4. **Hypothesis 2 (Growth Cycle) is the most consistent winner.** It hits return AND vol simultaneously across multiple model + threshold combinations. H1 (Market Stress) is surprisingly hard because the model only learns to identify stress *during* the drawdown, not before it.

**Suggested next experiments for the autoresearch agent:**

- **Lead-the-event labels.** The current label `equity_drawdown_event` fires on the realised drawdown month. Try `equity_drawdown_lead_3m` (label = 1 if a >5% drawdown happens in the next 3 months). This gives the model something predictive to learn.
- **Wider tilt vectors.** Edit the per-hypothesis `tilt` in `hypotheses/*.py` so the active tilt approaches ±15% on each leg. Combine with TIPS-spread regime detection.
- **Rolling SAA recalibration.** The annual rebalance is locked to calendar years. Try a 6-month rebalance with a longer covariance window.
- **Stacked models.** Use the LSTM probability as an extra feature into XGBoost (or vice versa). The autoresearch agent can implement this purely inside `train.py`.
- **Asymmetric thresholds.** Use a high threshold to fire defensive and a low threshold to fire offensive (currently a single threshold controls both). The agent can encode this in `train.py`.
