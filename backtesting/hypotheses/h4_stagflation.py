"""H4 — Stagflation hypothesis.

Signals: HY OAS, ISM PMI, VIX, TIPS-vs-Treasury performance, gold trend.

When growth weakens AND inflation hedges (TIPS, gold) outperform, tilt
towards inflation-protected exposures within IPS bands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.core.data import MarketPanel, monthly_resample
from backtesting.hypotheses.base import (
    Hypothesis,
    HypothesisMeta,
    momentum,
    realised_vol,
    stagflation_label,
)


class StagflationHypothesis(Hypothesis):
    meta = HypothesisMeta(
        name="h4",
        title="H4 — Stagflation Signal",
        summary=(
            "Growth weakens (HY OAS rising, ISM falling, VIX rising) while "
            "inflation hedges (TIPS, gold) outperform Treasuries. Tilt towards "
            "Bonds (proxy for TIPS) and Gold; keep crypto small for vol control."
        ),
        signal_when_active=-1,
        tilt={
            "US Equity": -0.10,
            "REITs": -0.05,
            "Bitcoin": -0.03,
            "US Bonds": +0.08,
            "Gold": +0.10,
            "JPY": +0.00,
        },
    )

    feature_columns = [
        "hy_oas_level",
        "hy_oas_chg_3m",
        "ism_level",
        "ism_chg_3m",
        "vix_level",
        "vix_chg_1m",
        "tips_minus_bond_3m",
        "tips_minus_bond_6m",
        "gold_mom_3m",
        "gold_mom_6m",
        "gold_minus_spx_6m",
    ]

    def compute_features(self, panel: MarketPanel) -> pd.DataFrame:
        signals = panel.signals
        prices = panel.prices

        hy = signals["hy_oas"]
        ism = signals["ism_pmi"]
        vix = signals["vix"]
        gold = prices["Gold"]
        spx = prices["US Equity"]
        bonds = prices["US Bonds"]
        tips = panel.tips

        feats = pd.DataFrame(index=signals.index)
        feats["hy_oas_level"] = hy
        feats["hy_oas_chg_3m"] = hy.diff(63)
        feats["ism_level"] = ism
        feats["ism_chg_3m"] = ism.diff(63)
        feats["vix_level"] = vix
        feats["vix_chg_1m"] = vix.diff(21)
        feats["tips_minus_bond_3m"] = momentum(tips, 63) - momentum(bonds, 63)
        feats["tips_minus_bond_6m"] = momentum(tips, 126) - momentum(bonds, 126)
        feats["gold_mom_3m"] = momentum(gold, 63)
        feats["gold_mom_6m"] = momentum(gold, 126)
        feats["gold_minus_spx_6m"] = momentum(gold, 126) - momentum(spx, 126)

        return monthly_resample(feats[self.feature_columns]).dropna()

    def compute_label(self, panel: MarketPanel) -> pd.Series:
        return stagflation_label(panel)
