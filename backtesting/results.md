# Whitmore Hybrid TAA Results

Generated from the rerun artifact set based on `Whitmore_IPS_extracted.txt` and `Whitmore_IPS.pdf`.

## IPS Targets

- Return p.a. >= 8%
- Volatility p.a. <= 15%
- Max drawdown >= -25%

## Best Current Run

- Best TAA: `h1/xgb_lstm`
- IPS pass count: `3/3`
- Return p.a.: `9.49%`
- Volatility p.a.: `9.13%`
- Max drawdown: `-19.05%`
- Artifact directory: `backtesting/artifacts/h1_xgb_lstm_83955db5`

## Same-Window Comparison

| Portfolio | IPS Pass | Return p.a. | Vol p.a. | Max DD | VaR 95 Daily | Sharpe rf=2% | Calmar |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Benchmark 1 60/40 | 1/3 | 6.50% | 10.54% | -32.93% | -0.93% | 0.462 | 0.197 |
| Benchmark 2 Diversified Policy | 1/3 | 7.71% | 11.91% | -35.24% | -1.03% | 0.516 | 0.219 |
| SAA | 2/3 | 8.50% | 10.41% | -30.54% | -0.93% | 0.645 | 0.278 |
| Best TAA (h1/xgb_lstm) | 3/3 | 9.49% | 9.13% | -19.05% | -0.85% | 0.820 | 0.498 |

## Top Rerun Trials

| Hypothesis | Model | Assets | IPS Pass | Objective | Return | Vol | Max DD | Signals | Threshold | Regime View | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h1 | xgb_lstm | ips_available | 3/3 | 18.698 | 9.49% | 9.13% | -19.05% | 61 | 0.50 | 2.00 | keep |
| h1 | xgb_lstm | policy_plus_bitcoin | 3/3 | 18.698 | 9.49% | 9.13% | -19.05% | 61 | 0.50 | 2.00 | discard |
| h1 | xgb_lstm | policy_plus_bitcoin | 3/3 | 19.002 | 9.24% | 9.19% | -19.05% | 53 | 0.55 | 2.00 | discard |
| h1 | xgb_lstm | ips_available | 3/3 | 19.002 | 9.24% | 9.19% | -19.05% | 53 | 0.55 | 2.00 | discard |
| h1 | xgb_lstm | ips_available | 3/3 | 19.731 | 9.73% | 9.29% | -20.17% | 61 | 0.50 | 1.50 | keep |
| h1 | xgb_lstm | policy_plus_bitcoin | 3/3 | 19.731 | 9.73% | 9.29% | -20.17% | 61 | 0.50 | 1.50 | discard |
| h1 | xgb_lstm | policy_plus_bitcoin | 3/3 | 19.947 | 9.55% | 9.33% | -20.17% | 53 | 0.55 | 1.50 | discard |
| h1 | xgb_lstm | ips_available | 3/3 | 19.947 | 9.55% | 9.33% | -20.17% | 53 | 0.55 | 1.50 | discard |
| h1 | xgb_lstm | ips_available | 3/3 | 21.312 | 9.57% | 9.51% | -21.37% | 61 | 0.50 | 1.00 | keep |
| h1 | xgb_lstm | policy_plus_bitcoin | 3/3 | 21.312 | 9.57% | 9.51% | -21.37% | 61 | 0.50 | 1.00 | discard |
| h1 | xgb_lstm | policy_plus_bitcoin | 3/3 | 21.472 | 9.43% | 9.53% | -21.37% | 53 | 0.55 | 1.00 | discard |
| h1 | xgb_lstm | ips_available | 3/3 | 21.472 | 9.43% | 9.53% | -21.37% | 53 | 0.55 | 1.00 | discard |
| h1 | xgb_lstm | policy_core | 2/3 | 1022.743 | 6.53% | 8.89% | -18.92% | 53 | 0.55 | 2.00 | discard |
| h1 | xgb_lstm | policy_core | 2/3 | 1022.910 | 6.51% | 8.79% | -19.14% | 61 | 0.50 | 2.00 | discard |
| h1 | xgb_lstm | policy_core | 2/3 | 1023.943 | 6.46% | 8.99% | -19.88% | 53 | 0.55 | 1.50 | discard |
| h1 | xgb_lstm | policy_core | 2/3 | 1024.714 | 6.35% | 8.91% | -20.50% | 61 | 0.50 | 1.50 | discard |
| h1 | xgb_lstm | policy_core | 2/3 | 1024.726 | 6.83% | 9.17% | -21.21% | 53 | 0.55 | 1.00 | discard |
| h1 | xgb_lstm | policy_core | 2/3 | 1024.909 | 6.71% | 9.11% | -21.21% | 61 | 0.50 | 1.00 | discard |
| h1 | xgb_lstm | policy_plus_bitcoin | 2/3 | 1026.600 | 9.79% | 10.05% | -25.67% | 61 | 0.50 | 0.50 | discard |
| h1 | xgb_lstm | ips_available | 2/3 | 1026.600 | 9.79% | 10.05% | -25.67% | 61 | 0.50 | 0.50 | keep |

## Benchmark Availability

| Asset | Benchmark 1 available | Benchmark 2 available |
| --- | --- | --- |
| Gold | 0.00% | 18.75% |
| US Equity | 60.00% | 50.00% |
| US REITs | 0.00% | 12.50% |
| US TIPS | 0.00% | 6.25% |
| US Treasuries | 40.00% | 12.50% |

Notes: US TIPS: BROAD_TIPS is unavailable; using local 0_5Y_TIPS_2002_D as the TIPS proxy. Japan Equity: Unavailable in local data; dropped from computable Benchmark 2 weights. China A-Shares: Unavailable in local data; dropped from computable Benchmark 2 weights. Silver: Unavailable in local data; dropped from computable Benchmark 2 weights. Swiss Franc: Unavailable in local data; dropped from computable Benchmark 2 weights.

The notebook `backtesting/hybrid_results_dashboard.ipynb` has been rerun in place with the updated benchmark and IPS lines.
