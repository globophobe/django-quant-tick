from collections.abc import Generator
from typing import Any

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


class PurgedKFold(TimeSeriesSplit):
    """Time-series cross-validation with purging and embargo.

    Uses TimeSeriesSplit for forward-only splits, then applies:
    - Purging: drop training samples whose event ends after test starts.
    - Embargo: drop training samples within N bars before test starts.
    """

    def __init__(self, n_splits: int = 5, embargo_bars: int = 96, **kwargs) -> None:
        """Initialize the purged time-series cross-validator."""
        super().__init__(n_splits=n_splits, **kwargs)
        self.embargo_bars = embargo_bars

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
        event_end_exclusive_idx: np.ndarray | None = None,
        bar_idx: np.ndarray | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate purged train/test splits."""
        if bar_idx is None:
            raise ValueError("bar_idx is required for PurgedKFold.split")
        if event_end_exclusive_idx is None:
            yield from super().split(X, y, groups)
            return

        for train_idx, test_idx in super().split(X, y, groups):
            test_bars = bar_idx[test_idx]
            test_start_bar = test_bars.min()

            train_bars = bar_idx[train_idx]
            train_event_ends = event_end_exclusive_idx[train_idx]

            purge_mask = train_event_ends <= test_start_bar

            if self.embargo_bars > 0:
                embargo_start = test_start_bar - self.embargo_bars
                embargo_mask = train_bars < embargo_start
                final_mask = purge_mask & embargo_mask
            else:
                final_mask = purge_mask

            purged_train_idx = train_idx[final_mask]

            if len(purged_train_idx) > 0:
                yield purged_train_idx, test_idx
