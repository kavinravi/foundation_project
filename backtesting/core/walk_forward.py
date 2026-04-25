"""Walk-forward fold construction.

Splits a monthly index into ``n_folds`` expanding-window train / test pairs
plus an embedded validation slice (last 12 months of each train block) for
hyper-parameter selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex

    def describe(self) -> str:
        return (
            f"Fold {self.fold_id}: train {self.train.min().date()}–{self.train.max().date()}"
            f" ({len(self.train)}m), val {self.validation.min().date()}–{self.validation.max().date()}"
            f" ({len(self.validation)}m), test {self.test.min().date()}–{self.test.max().date()}"
            f" ({len(self.test)}m)"
        )


def make_walk_forward_folds(
    monthly_index: pd.DatetimeIndex,
    *,
    initial_train_months: int = 84,
    test_months: int = 24,
    val_months: int = 12,
    n_folds: int = 4,
    step_months: int | None = None,
) -> list[Fold]:
    """Construct ``n_folds`` expanding-window folds.

    Each fold's train block grows by ``step_months`` and the test block stays
    fixed-size. ``step_months`` defaults to ``test_months`` (no overlap).
    """
    if step_months is None:
        step_months = test_months

    monthly_index = pd.DatetimeIndex(monthly_index).sort_values()
    folds: list[Fold] = []
    train_end = initial_train_months
    for fold_id in range(n_folds):
        if train_end + test_months > len(monthly_index):
            break
        train = monthly_index[:train_end]
        validation = train[-val_months:]
        train_for_fit = train[:-val_months]
        test = monthly_index[train_end : train_end + test_months]
        folds.append(
            Fold(
                fold_id=fold_id + 1,
                train=train_for_fit,
                validation=validation,
                test=test,
            )
        )
        train_end += step_months
    return folds


__all__ = ["Fold", "make_walk_forward_folds"]
