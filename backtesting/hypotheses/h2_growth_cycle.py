"""H2 — Growth cycle hypothesis.

Signals: ISM PMI (level + 3-month change), Conference Board confidence,
Umich sentiment, RTY/SPX relative performance.

When growth signals improve and small caps confirm, take more risk:
overweight Equity / REITs / Bitcoin within IPS bands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.core.data import MarketPanel, monthly_resample
from backtesting.hypotheses.base import (
    Hypothesis,
    HypothesisMeta,
    momentum,
    positive_growth_label,
)


class GrowthCycleHypothesis(Hypothesis):
    meta = HypothesisMeta(
        name="h2",
        title="H2 — Growth Cycle",
        summary=(
            "When ISM PMI, consumer confidence, and small-cap relative momentum "
            "are all improving, the economy is entering a favourable regime. "
            "Lean into risk assets while staying within IPS bands."
        ),
        signal_when_active=+1,  # risk-on
        tilt={
            "US Equity": +0.10,
            "US REITs": +0.05,
            "Bitcoin": +0.03,
            "US Treasuries": -0.08,
            "US TIPS": -0.03,
            "Gold": -0.04,
        },
    )

    feature_columns = [
        "ism_level",
        "ism_chg_3m",
        "ism_chg_6m",
        "cb_conf_level",
        "cb_conf_chg_3m",
        "umich_level",
        "umich_chg_3m",
        "rty_minus_spx_3m",
        "rty_mom_6m",
        "spx_mom_6m",
        "unemp_chg_6m",
    ]

    def compute_features(self, panel: MarketPanel) -> pd.DataFrame:
        signals = panel.signals
        prices = panel.prices

        ism = signals["ism_pmi"]
        cb = signals["cb_confidence"]
        umich = signals["umich_sentiment"]
        rty = signals["rty_index"]
        spx = prices["US Equity"]
        unemp = signals["unemployment"]

        feats = pd.DataFrame(index=signals.index)
        feats["ism_level"] = ism
        feats["ism_chg_3m"] = ism.diff(63)
        feats["ism_chg_6m"] = ism.diff(126)
        feats["cb_conf_level"] = cb
        feats["cb_conf_chg_3m"] = cb.diff(63)
        feats["umich_level"] = umich
        feats["umich_chg_3m"] = umich.diff(63)
        feats["rty_minus_spx_3m"] = momentum(rty, 63) - momentum(spx, 63)
        feats["rty_mom_6m"] = momentum(rty, 126)
        feats["spx_mom_6m"] = momentum(spx, 126)
        feats["unemp_chg_6m"] = unemp.diff(126)

        return monthly_resample(feats[self.feature_columns]).dropna()

    def compute_label(self, panel: MarketPanel) -> pd.Series:
        return positive_growth_label(panel, return_threshold=0.025, horizon_days=63)
