# Whitmore Hybrid TAA Results

Generated from the current autoresearch sweep after rebasing to `Whitmore_IPS.pdf`.

## IPS Targets

- Return p.a. >= 8%
- Volatility p.a. <= 15%
- Max drawdown >= -25%

## Best Current Run

- Best TAA: `h1/xgb_lstm`
- IPS pass count: `3/3`
- Return p.a.: `9.52%`
- Volatility p.a.: `9.11%`
- Max drawdown: `-18.72%`
- Settings: threshold `0.50`, TAA band `0.50`, risk aversion `4.0`, BL tau `0.05`, regime view scale `2.00`
- Artifact directory: `backtesting/artifacts/h1_xgb_lstm_82a11850`

## Same-Period Portfolio Comparison

| Portfolio | IPS Pass | Return p.a. | Vol p.a. | Max DD | VaR 95 Daily | Sharpe rf=2% | Calmar |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Benchmark 1 60/40 | 1/3 | 6.50% | 10.54% | -32.93% | -0.93% | 0.462 | 0.197 |
| Benchmark 2 Diversified | 1/3 | 7.71% | 11.91% | -35.24% | -1.03% | 0.516 | 0.219 |
| SAA | 2/3 | 8.50% | 10.41% | -30.54% | -0.93% | 0.645 | 0.278 |
| Best TAA (h1/xgb_lstm) | 3/3 | 9.52% | 9.11% | -18.72% | -0.85% | 0.824 | 0.509 |

## Top Autoresearch Trials

| Hypothesis | Model | Assets | IPS Pass | Objective | Return | Vol | Max DD | Signals | Threshold | Band | Regime View | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h1 | xgb_lstm | ips_available | 3/3 | 18.314 | 9.52% | 9.11% | -18.72% | 61 | 0.50 | 0.50 | 2.00 | keep |
| h1 | xgb_lstm | policy_plus_bitcoin | 3/3 | 18.698 | 9.49% | 9.13% | -19.05% | 61 | 0.50 | 0.50 | 2.00 | discard |
| h1 | xgb_lstm | ips_available | 3/3 | 18.705 | 9.22% | 9.17% | -18.75% | 53 | 0.55 | 0.50 | 2.00 | discard |
| h1 | xgb_lstm | policy_plus_bitcoin | 3/3 | 19.002 | 9.24% | 9.19% | -19.05% | 53 | 0.55 | 0.50 | 2.00 | discard |
| h1 | xgb_lstm | ips_available | 3/3 | 19.422 | 9.47% | 9.25% | -19.65% | 61 | 0.50 | 0.50 | 1.50 | keep |
| h1 | xgb_lstm | ips_available | 3/3 | 19.643 | 9.29% | 9.28% | -19.65% | 53 | 0.55 | 0.50 | 1.50 | discard |
| h1 | xgb_lstm | policy_plus_bitcoin | 3/3 | 19.731 | 9.73% | 9.29% | -20.17% | 61 | 0.50 | 0.50 | 1.50 | discard |
| h1 | xgb_lstm | policy_plus_bitcoin | 3/3 | 19.947 | 9.55% | 9.33% | -20.17% | 53 | 0.55 | 0.50 | 1.50 | discard |
| h1 | xgb_lstm | ips_available | 3/3 | 20.852 | 9.51% | 9.43% | -20.93% | 61 | 0.50 | 0.50 | 1.00 | keep |
| h1 | xgb_lstm | ips_available | 3/3 | 21.054 | 9.33% | 9.46% | -20.93% | 53 | 0.55 | 0.50 | 1.00 | discard |
| h1 | xgb_lstm | policy_plus_bitcoin | 3/3 | 21.312 | 9.57% | 9.51% | -21.37% | 61 | 0.50 | 0.50 | 1.00 | discard |
| h1 | xgb_lstm | policy_plus_bitcoin | 3/3 | 21.472 | 9.43% | 9.53% | -21.37% | 53 | 0.55 | 0.50 | 1.00 | discard |
| h1 | xgb_lstm | policy_plus_bitcoin | 2/3 | 1026.600 | 9.79% | 10.05% | -25.67% | 61 | 0.50 | 0.50 | 0.50 | keep |
| h1 | xgb_lstm | policy_plus_bitcoin | 2/3 | 1026.681 | 9.68% | 10.03% | -25.67% | 53 | 0.55 | 0.50 | 0.50 | discard |
| h1 | xgb_lstm | ips_available | 2/3 | 1026.815 | 9.50% | 9.98% | -25.67% | 61 | 0.50 | 0.75 | 0.50 | discard |
| h1 | xgb_lstm | ips_available | 2/3 | 1026.815 | 9.50% | 9.98% | -25.67% | 61 | 0.50 | 1.00 | 0.50 | discard |
| h1 | xgb_lstm | ips_available | 2/3 | 1026.815 | 9.50% | 9.98% | -25.67% | 61 | 0.50 | 0.50 | 0.50 | keep |
| h1 | xgb_lstm | ips_available | 2/3 | 1026.863 | 9.44% | 9.96% | -25.67% | 53 | 0.55 | 0.75 | 0.50 | discard |
| h1 | xgb_lstm | ips_available | 2/3 | 1026.863 | 9.44% | 9.96% | -25.67% | 53 | 0.55 | 0.50 | 0.50 | discard |
| h1 | xgb_lstm | ips_available | 2/3 | 1026.863 | 9.44% | 9.96% | -25.67% | 53 | 0.55 | 1.00 | 0.50 | discard |

## Benchmark 2 Availability

| Asset | Benchmark 1 available | Benchmark 2 available |
| --- | --- | --- |
| Gold | 0.00% | 18.75% |
| US Equity | 60.00% | 50.00% |
| US REITs | 0.00% | 12.50% |
| US TIPS | 0.00% | 6.25% |
| US Treasuries | 40.00% | 12.50% |

Notes: US TIPS: BROAD_TIPS is unavailable; using local 0_5Y_TIPS_2002_D as the TIPS proxy. Japan Equity: Unavailable in local data; dropped from computable Benchmark 2 weights. China A-Shares: Unavailable in local data; dropped from computable Benchmark 2 weights. Silver: Unavailable in local data; dropped from computable Benchmark 2 weights. Swiss Franc: Unavailable in local data; dropped from computable Benchmark 2 weights.

See `backtesting/hybrid_results_dashboard.ipynb` for charts: cumulative growth, IPS target bars, drawdowns, fold splits, regime probabilities, and asset weights over time.
