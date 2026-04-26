"""H3 — Two-stage regime change hypothesis.

Stage 1 (market): HY OAS, VIX, SPX trend.
Stage 2 (macro confirmation): ISM PMI, Conf. Board / Umich sentiment.

The model gets BOTH stage feature blocks. Empirically the two-stage
agreement should reduce false positives on the noisier stage-1 signals.
The training label is the same downside event used in H1, so the model
learns when stage-1 alarms are *true* positives in the data.
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


class TwoStageRegimeHypothesis(Hypothesis):
    meta = HypothesisMeta(
        name="h3",
        title="H3 — Two-Stage Regime Change",
        summary=(
            "Markets react first (HY OAS, VIX, SPX trend); macro confirms later "
            "(ISM PMI, sentiment). A model fed both blocks should require their "
            "joint agreement before triggering risk-off, improving precision."
        ),
        signal_when_active=-1,
        tilt={
            "US Equity": -0.12,
            "US REITs": -0.05,
            "Bitcoin": -0.03,
            "US Treasuries": +0.09,
            "US TIPS": +0.04,
            "Gold": +0.05,
            "JPY": +0.03,
        },
    )

    feature_columns = [
        # Stage 1 (market)
        "hy_oas_level",
        "hy_oas_chg_1m",
        "hy_oas_chg_3m",
        "vix_level",
        "vix_chg_1m",
        "spx_mom_3m",
        "spx_dd_6m",
        "spx_vol_3m",
        # Stage 2 (macro)
        "ism_level",
        "ism_chg_3m",
        "cb_conf_chg_3m",
        "umich_chg_3m",
        # Cross-stage interactions
        "stress_macro_interaction",
    ]

    def compute_features(self, panel: MarketPanel) -> pd.DataFrame:
        signals = panel.signals
        prices = panel.prices

        hy = signals["hy_oas"]
        vix = signals["vix"]
        spx = prices["US Equity"]
        ism = signals["ism_pmi"]
        cb = signals["cb_confidence"]
        umich = signals["umich_sentiment"]

        feats = pd.DataFrame(index=signals.index)
        feats["hy_oas_level"] = hy
        feats["hy_oas_chg_1m"] = hy.diff(21)
        feats["hy_oas_chg_3m"] = hy.diff(63)
        feats["vix_level"] = vix
        feats["vix_chg_1m"] = vix.diff(21)
        feats["spx_mom_3m"] = momentum(spx, 63)
        feats["spx_dd_6m"] = equity_drawdown(spx, 126)
        feats["spx_vol_3m"] = realised_vol(spx, 63)
        feats["ism_level"] = ism
        feats["ism_chg_3m"] = ism.diff(63)
        feats["cb_conf_chg_3m"] = cb.diff(63)
        feats["umich_chg_3m"] = umich.diff(63)

        # Hand-crafted interaction: stress AND deteriorating macro
        stress_score = (
            ((vix - vix.rolling(252).mean()) / vix.rolling(252).std()).clip(lower=0)
            + ((hy - hy.rolling(756).mean()) / hy.rolling(756).std()).clip(lower=0)
        )
        macro_drag = (-ism.diff(63)).clip(lower=0) + (-umich.diff(63)).clip(lower=0)
        feats["stress_macro_interaction"] = stress_score * macro_drag

        return monthly_resample(feats[self.feature_columns]).dropna()

    def compute_label(self, panel: MarketPanel) -> pd.Series:
        return equity_drawdown_event_label(panel, drawdown_threshold=-0.04, return_threshold=-0.02)
