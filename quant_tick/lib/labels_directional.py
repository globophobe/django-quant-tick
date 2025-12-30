import logging

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


def generate_directional_labels(
    df: DataFrame,
    H: int,
    vol_window: int = 48,
    k: float = 0.25,
) -> Series:
    """Generate directional labels based on forward returns.

    Args:
        df: DataFrame with 'close' and 'ret' columns
        H: Forward horizon in bars
        vol_window: Rolling window for volatility computation
        k: Volatility multiplier for threshold

    Returns:
        Series with labels {0: UP, 1: DOWN, 2: FLAT}
        Last H rows will be NaN (no forward data available)

    Algorithm:
        1. Compute r_H = close.shift(-H) / close - 1
        2. Compute sigma = ret.rolling(vol_window).std()
        3. Compute delta = k * sigma * sqrt(H)
        4. Assign UP (0) if r_H > delta, DOWN (1) if r_H < -delta, else FLAT (2)
    """
    r_H = df["close"].shift(-H) / df["close"] - 1

    sigma = df["ret"].rolling(window=vol_window).std()

    delta = k * sigma * np.sqrt(H)

    labels = np.full(len(df), 2, dtype=np.int8)
    labels[r_H > delta] = 0
    labels[r_H < -delta] = 1

    labels = labels.astype(float)
    labels[r_H.isna() | sigma.isna()] = np.nan

    return Series(labels, index=df.index, name=f"label_h{H}")


def label_stats(labels: Series) -> dict:
    """Compute label distribution statistics.

    Args:
        labels: Series with labels (0: UP, 1: DOWN, 2: FLAT), may contain NaN

    Returns:
        Dict with class distribution and sample count
    """
    labels_clean = labels.dropna()
    counts = labels_clean.value_counts()
    total = len(labels_clean)

    if total == 0:
        logger.warning("No valid labels found (all NaN)")
        return {"UP": 0.0, "DOWN": 0.0, "FLAT": 0.0, "n_samples": 0}

    dist = {
        "UP": counts.get(0, 0) / total,
        "DOWN": counts.get(1, 0) / total,
        "FLAT": counts.get(2, 0) / total,
        "n_samples": total,
    }

    max_pct = max(dist["UP"], dist["DOWN"], dist["FLAT"])
    if max_pct > 0.80:
        logger.warning(
            f"Extreme label imbalance: UP={dist['UP']:.1%}, "
            f"DOWN={dist['DOWN']:.1%}, FLAT={dist['FLAT']:.1%}"
        )

    return dist
