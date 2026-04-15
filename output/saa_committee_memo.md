# Assignment 4 Strategic Baseline

## Executive takeaway
The refreshed Zion pull confirms that the investable series extend far beyond the truncated 2020 export. The full six-asset overlap begins in `2010-07-19` because Bitcoin starts in mid-2010, while the staggered-availability backtest becomes feasible earlier in `2003` once REITs enter the universe and the IPS cap structure can be satisfied.

Using the latest `252d` covariance sample, the refreshed SAA recommendation is:

| Sleeve | Weight |
|---|---:|
| US Equity | 30.0% |
| US Bonds | 30.0% |
| REITs | 15.0% |
| Gold | 0.0% |
| Bitcoin | 0.0% |
| JPY | 25.0% |

## Updated Assignment 4 observations
- The strategic portfolio remains long-only, fully invested, and inside the tighter `25%` non-traditional cap.
- The `252d` setup is still the cleanest committee baseline because it balances recency against parameter noise.
- The best ex-post Calmar ratio in the window sweep was `84d`, but the one-year setup remains more defensible for the TAA benchmark.

## Risk context
- Refreshed SAA CAGR: `5.1%`.
- Refreshed SAA volatility: `8.2%`.
- Refreshed SAA max drawdown: `-26.3%`.
- Highest correlation regime ended on `2023-01-08` with average pairwise correlation `0.59`.

## Why this baseline matters for Assignment 4
The tactical overlay is not competing with a 60/40 benchmark. It is competing against this refreshed strategic allocation, built from the full Zion history rather than the truncated CSV shortcut.
