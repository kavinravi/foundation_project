# autoresearch — Whitmore TAA edition

This is the project-specific instruction file for an autonomous research
agent (Claude/Codex) running on top of Karpathy's `autoresearch` loop:
<https://github.com/karpathy/autoresearch>. The loop is the same: the
agent edits **one** file, runs it, checks a single scalar metric, and
keeps or reverts. Here that file is `backtesting/train.py` and the
metric is `ips_loss` (lower is better).

## Setup (do once at the start of an autoresearch session)

1. Agree a run tag with the user (e.g. `apr24-h1-lstm`). The branch
   `autoresearch/<tag>` must not already exist.
2. `git checkout -b autoresearch/<tag>` from `master`.
3. Read these files in full for context:
   - `backtesting/program.md` (this file)
   - `backtesting/train.py` — the only file you edit
   - `backtesting/prepare.py` — locked evaluation harness, read-only
   - `backtesting/hypotheses/h?_*.py` — for the hypothesis you are
     focusing on. You **may** also edit a hypothesis file to add or
     change features (treat it as part of "the experiment file").
   - `backtesting/models/<model>_model.py` — for the model you are
     focusing on. You may edit it (architecture / training loop tweaks),
     but keep the `BaseModel` interface intact.
4. Initialise (or check) `results.tsv`. The baseline row was written by
   `run_all_baselines.py`; your additions go below.
5. Confirm with the user, then start the loop.

## What you can edit

- `backtesting/train.py` — `HYPOTHESIS`, `MODEL`, `SIGNAL_THRESHOLD`,
  `SEQUENCE_LENGTH`, `HYPERPARAMETERS`. This is the main lever.
- `backtesting/hypotheses/h?_*.py` — feature engineering and the
  `tilt` vector. Adding/dropping features is fair game. Changing the
  binary label definition is also fair game (e.g. tighten the downside
  threshold for H1) but record it explicitly in the TSV description.
- `backtesting/models/<model>_model.py` — model architecture, training
  loop, regularisation, optimiser. Keep the `BaseModel.predict_proba`
  contract intact (must return a `pd.Series` indexed identically to
  the input frame and bounded in [0, 1]).

## What you may NOT edit

- `backtesting/prepare.py` — locked. Contains the walk-forward
  schedule and the `ips_loss` evaluation. Do not change it; the metric
  must stay comparable across all experiments.
- `backtesting/core/ips.py` — the IPS constants and the loss
  formulation. Do not change them.
- `backtesting/run_all_baselines.py` — only re-run, do not edit.
- The asset universe (6 Zion assets) — fixed by the IPS.

## What success looks like

The IPS has three binding constraints:

| Constraint | Target | Baseline SAA | Best Baseline TAA | Gap to close |
|---|---|---|---|---|
| Return p.a. | ≥ 5% | 4.68% | varies | small (~0.3 pp) |
| Volatility p.a. | ≤ 11% | 6.15% | varies | already met |
| Max drawdown | ≥ -13% | -19.88% | varies | **large (~6.9 pp)** |

Drawdown is by far the hardest constraint. Until you find a TAA that
materially shrinks the 2008-2009 drawdown, you will be stuck at 1/3 or
2/3 IPS. The composite metric is:

```
ips_loss = 100 * ( max(0, 5% - return_pa)
                 + max(0, vol_pa - 11%)
                 + max(0, -13% - max_dd) )
         - 0.10 * sharpe_rf2
```

A feasible portfolio (3/3 IPS) gets a non-positive loss whose magnitude
is the Sharpe tiebreaker. An infeasible portfolio pays a linear penalty
in percentage points missed.

## How to run an experiment

```bash
python -m backtesting.train --hypothesis h1 --model lstm \
    --threshold 0.6 --seq-len 18 \
    --description "lstm h1 thr0.6 seq18 hidden128"
```

Or by editing the constants at the top of `train.py` and just running
`python -m backtesting.train`. The script prints a metric block ending
in `ips_loss: <value>` which is your scalar.

Wall-clock budget per experiment: roughly **30 seconds** for XGBoost,
**60 seconds** for LSTM/Transformer at default sizes on the RTX 5090.
If a run exceeds 5 minutes, kill it and treat it as a failure.

## Logging

Append one line per experiment to `backtesting/results.tsv`:

```
timestamp\thypothesis\tmodel\tthreshold\tips_loss\tips_pass\tdelta_dd\tdelta_ret\tstatus\tdescription
```

Statuses: `keep` (improved on the branch best), `discard` (worse), or
`crash`. Description is a short human-readable summary of what you tried.

## The experiment loop

```
LOOP FOREVER:
  1. Pick the next change to try (architecture / hyperparameter /
     feature / threshold).
  2. Edit train.py (and possibly the hypothesis or model file).
  3. git commit -am "<short description>"
  4. python -m backtesting.train > run.log 2>&1
  5. Read out the metric: grep "^ips_loss:" run.log
  6. If empty, the run crashed: tail -n 50 run.log, attempt a fix,
     give up after a few attempts.
  7. Append to results.tsv.
  8. If ips_loss improved (lower), keep the commit. Otherwise revert
     (`git reset --hard HEAD~1`) and try a different idea.
```

## Suggestions for productive things to try

- **Threshold sweep** — the cheapest win. Try 0.40, 0.45, 0.50, 0.55,
  0.60, 0.65 for each (hypothesis, model). Higher threshold = signal
  fires less often, which tends to lower drawdown of the TAA portfolio
  (less noise) but can also lower return.
- **Sequence length** for LSTM/Transformer — try 6, 12, 18, 24 months.
- **Feature engineering** for H1 — try VIX term structure (if you can
  source it), HY OAS rate of change at multiple horizons, or a
  composite `stress_score` feature that combines several signals.
- **Bigger tilt** for H1/H3 stress regimes — the current tilt is
  conservative; the IPS allows ±15% per asset. Try tilting US Equity
  by -15% and US Bonds by +15% when stress fires.
- **Lower SAA drawdown** — the SAA itself is at -19.9%. Try a smaller
  US Equity floor (note: 20% is the IPS floor and is binding, so this
  may not be possible without an IPS amendment).
- **Walk-forward refresh policy** — try retraining more frequently
  (per-month rolling fit instead of per-fold).
- **Class re-weighting** — increase `pos_weight` so the LSTM /
  Transformer pay more attention to rare stress months.

Good luck. Don't ask "should I keep going?" — keep iterating until the
human stops you. Aim for `ips_pass = 3/3` with the highest Sharpe you
can find.
