# autoresearch - Whitmore hybrid TAA edition

Project-specific instructions for running an autonomous research loop in the
style of Andrej Karpathy's `autoresearch` repo. The loop edits code, runs one
experiment, reads one scalar, keeps improvements, and retains trial disclosure.

The scalar is `ips_objective` from `backtesting/core/ips.py`; lower is better.
The objective is constraint-first and is based only on the three binding
Whitmore IPS acceptance constraints from `Whitmore_IPS.pdf`:

- Annualized return must be at least 8%.
- Annualized volatility must be 15% or lower.
- Maximum drawdown must be no worse than 25%, encoded as `max_drawdown >= -0.25`.

Do not optimize validation accuracy, BCE loss, MSE, Sharpe, or Calmar directly.
Those are diagnostics. The autoresearch objective is to reach `ips_pass = 3/3`
and then improve return, volatility, and drawdown inside the feasible region.

## Required Research Design

Every valid run must test the assignment's four hypotheses from
`hypotheses.txt`:

- `h1` Market Stress
- `h2` Growth Cycle
- `h3` Two-Stage Regime Change
- `h4` Stagflation

Every valid TAA run must use the hybrid stack:

- XGBoost is the signal/regime classifier.
- LSTM or Transformer is the multi-asset next-month return forecaster.
- Forecasted returns become Black-Litterman absolute views.
- The final portfolio is solved with a Markowitz mean-variance optimizer.
- SAA remains the baseline allocation; TAA can only move within the IPS TAA
  bands and must pay transaction costs on turnover.

Single-model experiments such as `--model xgb`, `--model lstm`, or
`--model transformer` are invalid for this assignment unless used only as
diagnostics. The experiment model label should be `xgb_lstm` or
`xgb_transformer`.

## Whitmore IPS Portfolio Rules

The implemented local tradable universe is the intersection of the Whitmore IPS
and local `data/` availability:

- `US Equity` (`SPXT`) - Core
- `US Treasuries` (`LBUSTRUU`) - Core
- `US TIPS` (`0_5Y_TIPS_2002_D`) - Core proxy because `BROAD_TIPS` is absent
- `US REITs` (`B3REITT`) - Satellite
- `Gold` (`XAU`) - Satellite
- `Bitcoin` (`XBTUSD`) - Non-Traditional
- `JPY` (`USDJPY`, inverted for USD-investor returns) - Opportunistic

Do not fabricate missing IPS series. Missing local series include UK equity,
Japan equity, China A-shares, silver, and Swiss franc. Benchmark 2 is computed
from available components and renormalized to 100%; the omitted weights must be
disclosed in charts and notes.

Binding constraints:

- Long-only and fully invested.
- SAA rebalances annually on the last trading day of the calendar year.
- Transaction costs are 5 bps round trip on all turnover.
- Core allocation must be at least 40%.
- Satellite allocation must be no more than 45%.
- Non-Traditional allocation must be no more than 20% after the 2026 amendment.
- Opportunistic allocation must be no more than 15%.
- Opportunistic single-asset exposure must be no more than 5%.
- Single-sleeve/asset exposure must be no more than 45%.

SAA policy bands for available assets:

- `US Equity`: 30% to 45%, target 40%.
- `US Treasuries`: 5% to 15%, target 10%.
- `US TIPS`: 0% to 10%, target 5%.
- `US REITs`: 5% to 20%, target 10%.
- `Gold`: 10% to 25%, target 15%.
- `Bitcoin`: 0% to 5%, target 0%.
- `JPY`: 0% target and 0% SAA max; TAA-only opportunistic asset.

TAA bands for available assets:

- `US Equity`: 20% to 45%.
- `US Treasuries`: 0% to 35%.
- `US TIPS`: 0% to 25%.
- `US REITs`: 0% to 25%.
- `Gold`: 0% to 30%.
- `Bitcoin`: 0% to 10%.
- `JPY`: 0% to 5% as an opportunistic asset.

## Benchmarks

Both Whitmore benchmarks must be computed and shown in the results:

- Benchmark 1 Traditional 60/40: 60% `US Equity`, 40% `US Treasuries`.
- Benchmark 2 Diversified Policy: 40% `US Equity`, 5% Japan Equity, 5% China
  A-shares, 10% `US Treasuries`, 5% `US TIPS`, 10% `US REITs`, 15% `Gold`,
  5% Silver, 5% Swiss Franc.

Because Japan equity, China A-shares, silver, and Swiss franc are not present in
local `data/`, Benchmark 2 is computed from the available components and
renormalized. The computable Benchmark 2 weights are therefore:

- `US Equity`: 50.00%
- `US Treasuries`: 12.50%
- `US TIPS`: 6.25%
- `US REITs`: 12.50%
- `Gold`: 18.75%

## Data Rules

- Anything in `data/` may be used as a tradable asset only if it is allowed by
  the IPS universe or a documented IPS proxy/opportunistic sleeve asset.
- Anything in `data/signals/` may be used as a signal/indicator.
- Additional derived indicators are allowed if they are point-in-time and built
  only from data available at the decision date.
- `data/signals/` series are indicator inputs only unless the same asset also
  exists as a tradable price/return series under `data/`.
- Do not import or use `build_assignment_4.py`; it is a historical baseline.

## Setup At Session Start

1. Read `Assignment.txt`, `hypotheses.txt`, `Whitmore_IPS.pdf`, and this file.
2. Delete stale generated result files before restarting the loop:
   `backtesting/results.*`, `backtesting/artifacts/*`, generated notebooks, and
   files in `output/` whose names contain result/report/chart/table data.
3. Confirm `backtesting/train.py` runs the hybrid path:
   XGB classifier plus `--return-model lstm|transformer`.
4. Confirm `backtesting/prepare.py` scores only the IPS objective and uses the
   fixed walk-forward folds.
5. Run at least one smoke test before starting a larger sweep:

```bash
python -m backtesting.train --hypothesis h1 --return-model lstm \
    --threshold 0.55 --seq-len 6 \
    --assets "US Equity" "US Treasuries" "US TIPS" "US REITs" "Gold" "Bitcoin" "JPY" \
    --taa-band 0.50 --description "smoke Whitmore hybrid xgb+lstm"
```

## What You Can Edit

- `backtesting/train.py`: main experiment knobs, XGB hyperparameters, neural
  return-forecast hyperparameters, active assets, thresholds, sequence length,
  TAA band, BL tau/confidence, forecast scaling, and Markowitz risk aversion.
- `backtesting/hypotheses/h?_*.py`: features, labels, and regime view tilts.
  Any change must remain point-in-time and must be recorded in the results
  description.
- `backtesting/models/return_forecaster.py`: LSTM/Transformer return
  forecasting architecture and training loop.
- `backtesting/models/xgb_model.py`: XGB classifier settings only.
- `backtesting/core/backtest.py`: portfolio construction only when changing the
  Black-Litterman, Markowitz, or IPS constraint implementation.

## What You May Not Edit

- The IPS targets and hard constraints in `backtesting/core/ips.py` unless the
  source IPS document is reread and the current encoding is shown to be wrong.
- `backtesting/core/walk_forward.py`: fold construction, unless the assignment
  explicitly requires a different walk-forward schedule.
- Result artifacts as inputs to a new run.
- Any code path that reintroduces `build_assignment_4.py`.

## Valid Experiment Command

```bash
python -m backtesting.train --hypothesis h1 --return-model transformer \
    --threshold 0.55 --seq-len 12 \
    --assets "US Equity" "US Treasuries" "US TIPS" "US REITs" "Gold" "Bitcoin" "JPY" \
    --taa-band 0.50 \
    --risk-aversion 4.0 --bl-tau 0.05 \
    --forecast-scale 12.0 --regime-view-scale 0.50 \
    --description "h1 xgb+transformer BL/MV Whitmore baseline"
```

The script must print a metric block ending in:

```text
ips_objective: <value>
```

That value is the only scalar used for keep/revert decisions.

## Logging

Each run appends one row to `backtesting/results.tsv`:

```text
timestamp  hypothesis  model  threshold  ips_objective  ips_pass  delta_dd  delta_ret  status  description
```

Use statuses:

- `keep`: best objective so far for that hypothesis and hybrid model.
- `discard`: worse than the current best, retained only for trial disclosure.
- `crash`: failed run; include the failure mode in the description.
- `search`: deterministic sweep row.

Descriptions must include the return model, active assets, threshold, TAA band,
BL settings, and any feature/label changes.

## Autoresearch Loop

```text
LOOP:
  1. Pick one concrete change: threshold, assets, TAA band, BL confidence,
     risk aversion, sequence length, neural architecture, XGB settings,
     features, labels, or regime view scaling.
  2. Edit the smallest necessary file set.
  3. Run one hybrid experiment and capture `ips_objective`.
  4. Append the result to `results.tsv`.
  5. If the run improves the best objective for the same hypothesis and
     hybrid model, keep it.
  6. If it is worse, mark it `discard`; do not use `git reset --hard`.
  7. Move to the next hypothesis/model pair until all four hypotheses have
     tested both `xgb_lstm` and `xgb_transformer`.
```

Do not stop after one promising hypothesis. The assignment requires honest
trial disclosure across all four hypotheses.

## Productive Search Order

Start cheap and broaden only after the hybrid path is stable:

- Sweep thresholds: `0.45`, `0.50`, `0.55`, `0.60`, `0.65`.
- Sweep TAA bands: `0.25`, `0.50`, `0.75`, `1.00`.
- Sweep active asset setups using only IPS-compliant local data:
  `ips_available`, `policy_core`, `policy_plus_bitcoin`, and `policy_plus_jpy`.
- Sweep sequence lengths for neural return forecasts: `6`, `12`, `18`, `24`.
- Sweep BL controls: `risk_aversion` from `2` to `8`, `bl_tau` from `0.02` to
  `0.20`, and `regime_view_scale` from `0.25` to `1.00`.
- Improve labels so they lead the event instead of identifying the event during
  or after the drawdown.
- Add point-in-time features from `data/signals/` and defensible derived
  indicators such as spread changes, trend, realized volatility, relative
  momentum, and inflation-hedge relative strength.

The target is not merely "better than SAA". The target is `ips_pass = 3/3` for
the SAA plus TAA portfolio: at least 8% annual return, no more than 15%
annual volatility, and no worse than 25% maximum drawdown.
