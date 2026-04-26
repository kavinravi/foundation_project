"""H1 — Market stress hypothesis.

Signals: HY OAS, VIX, SPX trend, RTY/SPX relative performance.

When stress rises (model probability ≥ threshold), tilt away from risk
assets (Equity, REITs, Bitcoin) and towards defensive assets (Treasuries,
TIPS, Gold, JPY). The framework enforces the IPS bands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.core.data import MarketPanel, monthly_resample
from backtesting.hypotheses.base import (
    Hypothesis,
    HypothesisMeta,
    equity_drawdown,
    equity_drawdown_event_label,
    momentum,
    realised_vol,
)


class MarketStressHypothesis(Hypothesis):
    meta = HypothesisMeta(
        name="h1",
        title="H1 — Market Stress",
        summary=(
            "Credit spreads, implied vol, equity trend, and small-cap relative "
            "performance jointly mark deteriorating financial conditions. When "
            "the signal fires, tilt towards defensive assets within IPS bands."
        ),
        signal_when_active=-1,  # risk-off
        tilt={
            "US Equity": -0.10,
            "US REITs": -0.05,
            "Bitcoin": -0.03,
            "US Treasuries": +0.08,
            "US TIPS": +0.04,
            "Gold": +0.05,
            "JPY": +0.03,
        },
    )

    feature_columns = [
        "hy_oas_level",
        "hy_oas_chg_3m",
        "hy_oas_z_3y",
        "vix_level",
        "vix_chg_1m",
        "vix_z_1y",
        "spx_mom_3m",
        "spx_dd_6m",
        "spx_vol_3m",
        "rty_minus_spx_3m",
        "agg_oas_chg_3m",
    ]

    def compute_features(self, panel: MarketPanel) -> pd.DataFrame:
        signals = panel.signals
        prices = panel.prices

        hy_oas = signals["hy_oas"]
        vix = signals["vix"]
        agg_oas = signals["agg_oas"]
        spx = prices["US Equity"]
        rty = signals["rty_index"]

        feats = pd.DataFrame(index=signals.index)
        feats["hy_oas_level"] = hy_oas
        feats["hy_oas_chg_3m"] = hy_oas.diff(63)
        feats["hy_oas_z_3y"] = (hy_oas - hy_oas.rolling(756).mean()) / hy_oas.rolling(756).std()
        feats["vix_level"] = vix
        feats["vix_chg_1m"] = vix.diff(21)
        feats["vix_z_1y"] = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
        feats["spx_mom_3m"] = momentum(spx, 63)
        feats["spx_dd_6m"] = equity_drawdown(spx, 126)
        feats["spx_vol_3m"] = realised_vol(spx, 63)
        feats["rty_minus_spx_3m"] = momentum(rty, 63) - momentum(spx, 63)
        feats["agg_oas_chg_3m"] = agg_oas.diff(63)

        return monthly_resample(feats[self.feature_columns]).dropna()

    def compute_label(self, panel: MarketPanel) -> pd.Series:
        return equity_drawdown_event_label(panel, drawdown_threshold=-0.04, return_threshold=-0.02)
