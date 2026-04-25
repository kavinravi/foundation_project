"""Hypothesis registry."""

from backtesting.hypotheses.base import Hypothesis
from backtesting.hypotheses.h1_market_stress import MarketStressHypothesis
from backtesting.hypotheses.h2_growth_cycle import GrowthCycleHypothesis
from backtesting.hypotheses.h3_two_stage import TwoStageRegimeHypothesis
from backtesting.hypotheses.h4_stagflation import StagflationHypothesis

HYPOTHESIS_REGISTRY: dict[str, type[Hypothesis]] = {
    "h1": MarketStressHypothesis,
    "h2": GrowthCycleHypothesis,
    "h3": TwoStageRegimeHypothesis,
    "h4": StagflationHypothesis,
}


def build_hypothesis(name: str) -> Hypothesis:
    if name not in HYPOTHESIS_REGISTRY:
        raise KeyError(f"Unknown hypothesis '{name}'. Choose from {sorted(HYPOTHESIS_REGISTRY)}.")
    return HYPOTHESIS_REGISTRY[name]()


__all__ = [
    "GrowthCycleHypothesis",
    "HYPOTHESIS_REGISTRY",
    "Hypothesis",
    "MarketStressHypothesis",
    "StagflationHypothesis",
    "TwoStageRegimeHypothesis",
    "build_hypothesis",
]
