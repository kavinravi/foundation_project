# Whitmore TAA Backtesting Framework

End-to-end pipeline for testing the four TAA hypotheses
(`hypotheses.txt`) against the Whitmore IPS constraints, using XGBoost as
the signal classifier plus LSTM/Transformer return forecasters feeding a
Black-Litterman / Markowitz portfolio layer.

The framework is explicitly modelled on Karpathy's
[`autoresearch`](https://github.com/karpathy/autoresearch) so an
autonomous coding agent (Claude / Codex) can iterate overnight on
`train.py` to improve the IPS objective — see `program.md`.

## Layout

```
backtesting/
├── core/
│   ├── ips.py            IPS targets + objective / scorecard helpers
│   ├── data.py           Unified market panel + candidate tradable assets
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
│   ├── xgb_model.py      XGBoost wrapper (GPU-first when CUDA is available)
│   ├── lstm_model.py     PyTorch LSTM (RTX 5090 / Blackwell)
│   └── transformer_model.py  PyTorch Transformer encoder
├── prepare.py            Locked evaluation harness (do not edit)
├── train.py              Single-file experiment entry point (agent edits this)
├── run_threshold_search.py  Run the compact hybrid autoresearch sweep
├── program.md            autoresearch agent instructions
├── results.tsv           Append-only experiment log
└── artifacts/            Per-run JSON / CSV outputs
```

## Quick start

```bash
# Create an environment with a Blackwell-capable PyTorch build.
# In this workspace the verified combo is torch 2.9.1+cu128 + xgboost 3.2.0.
# If your stable PyTorch channel does not yet support the RTX 5090 on your
# driver stack, use the matching nightly CUDA wheel instead.

# Core deps
pip install --user xgboost==3.2.0 statsmodels reportlab scikit-optimize scipy pandas numpy

# PyTorch: use a CUDA build that supports your 5090. Keep this project
# PyTorch-only; do not add TensorFlow/Keras.
# Example when nightly/cu128 is needed:
# pip install --pre --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Smoke-test a single hybrid combo
python -m backtesting.train --hypothesis h1 --return-model lstm \
  --threshold 0.55 --seq-len 6 \
  --assets "US Equity" "US Treasuries" "US TIPS" "US REITs" "Gold" "Bitcoin" \
  --taa-band 0.50

# Run all four hypotheses with both hybrid model stacks
python -m backtesting.run_threshold_search
```

## Running the autoresearch agent

Open this directory in Cursor / Claude Code with full edit permissions
disabled, then prompt:

> Hi, have a look at `backtesting/program.md` and let's kick off a new
> experiment! Let's do the setup first.

The agent will read `program.md`, branch off, and iterate on
`train.py` (and the hypothesis / model files) overnight to drive
`ips_objective` down.

## Runtime notes

- `backtesting/` is intentionally decoupled from `build_assignment_4.py`.
- Tradable candidates come from `data/`. `data/signals/` is indicator-only.
- XGBoost defaults to `device="cuda"` when CUDA is available.
- The PyTorch models enable TF32-friendly matmul/cuDNN settings on GPU.
- The autoresearch scalar is `ips_objective`, built only from annualized
  return, annualized volatility, and maximum drawdown.

## Reproducing a published result

Each kept experiment writes a JSON summary, the OOS signal series, the XGB
probabilities, neural return forecasts, TAA daily returns, and rebalance
weights to `backtesting/artifacts/<hypothesis>_<model>_<asset-hash>/`.
Re-run with the same seed and hyperparameters to reproduce.
