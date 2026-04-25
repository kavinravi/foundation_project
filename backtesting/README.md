# Whitmore TAA Backtesting Framework

End-to-end pipeline for testing the four TAA hypotheses
(`hypotheses.txt`) against the Whitmore IPS constraints, modelled by
XGBoost, LSTM, and Transformer classifiers.

The framework is explicitly modelled on Karpathy's
[`autoresearch`](https://github.com/karpathy/autoresearch) so an
autonomous coding agent (Claude / Codex) can iterate overnight on
`train.py` to improve the IPS-loss metric — see `program.md`.

## Layout

```
backtesting/
├── core/
│   ├── ips.py            IPS constants + loss / scorecard helpers
│   ├── data.py           Unified market panel: prices + signals + TIPS
│   ├── backtest.py       SAA, benchmark, TAA backtest engines
│   └── walk_forward.py   Expanding-window fold construction
├── hypotheses/
│   ├── base.py           Hypothesis interface + shared label / feature helpers
│   ├── h1_market_stress.py
│   ├── h2_growth_cycle.py
│   ├── h3_two_stage.py
│   └── h4_stagflation.py
├── models/
│   ├── base.py           BaseModel interface + standardisation utilities
│   ├── xgb_model.py      XGBoost wrapper
│   ├── lstm_model.py     PyTorch LSTM (RTX 5090 / Blackwell sm_120)
│   └── transformer_model.py  PyTorch Transformer encoder
├── prepare.py            Locked evaluation harness (do not edit)
├── train.py              Single-file experiment entry point (agent edits this)
├── run_all_baselines.py  Run all 4×3 = 12 combos and emit results.md
├── program.md            autoresearch agent instructions
├── results.md            Auto-generated baseline results
├── results.tsv           Append-only experiment log
└── artifacts/            Per-run JSON / CSV outputs
```

## Quick start

```bash
# (One-time) install Python deps not already present
pip install --user xgboost statsmodels reportlab scikit-optimize

# Smoke-test a single combo
python -m backtesting.train --hypothesis h1 --model xgb

# Run all 12 baselines and refresh results.md
python -m backtesting.run_all_baselines

# Or focus on one hypothesis
python -m backtesting.run_all_baselines --hypotheses h1 --models xgb lstm transformer
```

## Running the autoresearch agent

Open this directory in Cursor / Claude Code with full edit permissions
disabled, then prompt:

> Hi, have a look at `backtesting/program.md` and let's kick off a new
> experiment! Let's do the setup first.

The agent will read `program.md`, branch off, and iterate on
`train.py` (and the hypothesis / model files) overnight to drive
`ips_loss` down.

## Reproducing a published result

Each experiment writes a JSON summary, the OOS signal series, the TAA
daily returns, and the rebalance weights to
`backtesting/artifacts/<hypothesis>_<model>/`. Re-run with the same
seed and hyperparameters to reproduce.
