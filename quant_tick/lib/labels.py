import hashlib
import logging

import numpy as np
import pandas as pd

from quant_tick.lib.ml import (
    DEFAULT_ASYMMETRIES,
    DEFAULT_WIDTHS,
    generate_multi_config_labels,
)
from quant_tick.models import Candle, MLConfig, MLFeatureData, Symbol

logger = logging.getLogger(__name__)


def generate_labels(
    candle: Candle,
    symbol: Symbol,
    horizon_bars: int = 60,
    min_bars: int = 1000,
    widths: list[float] | None = None,
    asymmetries: list[float] | None = None,
) -> MLConfig | None:
    """Generate ML labels and feature data.

    Args:
        candle: Candle to generate labels for
        symbol: Symbol for position tracking
        horizon_bars: Prediction horizon in bars
        min_bars: Minimum bars required
        widths: Range widths (default: DEFAULT_WIDTHS)
        asymmetries: Asymmetries (default: DEFAULT_ASYMMETRIES)

    Returns:
        MLConfig if successful, None otherwise
    """
    widths = widths or DEFAULT_WIDTHS
    asymmetries = asymmetries or DEFAULT_ASYMMETRIES

    # Load candle data
    df = candle.get_candle_data()
    if df is None or len(df) < min_bars:
        n = len(df) if df is not None else 0
        logger.error(f"{candle}: insufficient data ({n}/{min_bars})")
        return None

    logger.info(f"{candle}: loaded {len(df)} bars with {len(df.columns)} columns")

    # Compute derived features from per-exchange columns
    df = _compute_features(df)

    # Generate labels
    labeled = generate_multi_config_labels(
        df,
        widths=widths,
        asymmetries=asymmetries,
        horizon_bars=horizon_bars,
    )

    n_configs = len(widths) * len(asymmetries)
    logger.info(
        f"{candle}: generated {len(labeled)} rows "
        f"({len(df)} bars x {n_configs} configs)"
    )

    # Compute schema hash
    schema_cols = sorted(labeled.columns.tolist())
    schema_hash = hashlib.sha256(",".join(schema_cols).encode()).hexdigest()[:16]

    # Get or create MLConfig
    config, created = MLConfig.objects.get_or_create(
        candle=candle,
        symbol=symbol,
        defaults={
            "horizon_bars": horizon_bars,
            "json_data": {
                "widths": widths,
                "asymmetries": asymmetries,
            },
        },
    )
    if created:
        logger.info(f"Created MLConfig for {candle}")

    # Save to MLFeatureData
    timestamp_from = df["timestamp"].min()
    timestamp_to = df["timestamp"].max()

    feature_data, created = MLFeatureData.objects.get_or_create(
        candle=candle,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        defaults={"schema_hash": schema_hash},
    )

    feature_data.file_data = MLFeatureData.prepare_data(labeled)
    feature_data.schema_hash = schema_hash
    feature_data.save()

    action = "Created" if created else "Updated"
    logger.info(
        f"{action} MLFeatureData: {timestamp_from} to {timestamp_to}, "
        f"schema={schema_hash}"
    )

    return config


def _is_multi_exchange(df: pd.DataFrame) -> bool:
    """Check if dataframe has per-exchange columns like binanceClose, coinbaseClose."""
    return any(c.endswith("Close") and c != "close" for c in df.columns)


def _compute_single_exchange_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for single-exchange candle data."""
    result = df.copy()
    if "close" not in df.columns:
        logger.warning("No close column found in candle data")
        return result

    ret = df["close"].pct_change()
    result["realizedVol"] = ret.rolling(20).std()
    result["realizedVol5"] = ret.rolling(5).std()
    return result


def _compute_multi_exchange_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for multi-exchange candle data."""
    result = df.copy()

    close_cols = [c for c in df.columns if c.endswith("Close") and c != "close"]
    canonical_col = close_cols[0]
    canonical_name = canonical_col.replace("Close", "")
    canonical_close = df[canonical_col]

    if "close" not in df.columns:
        result["close"] = canonical_close

    canonical_ret = canonical_close.pct_change()
    result[f"{canonical_name}Ret"] = canonical_ret

    for close_col in close_cols[1:]:
        other_name = close_col.replace("Close", "")
        other_close = df[close_col]

        result[f"{other_name}Missing"] = other_close.isna().astype(int)
        result[f"basis{other_name.title()}"] = other_close - canonical_close
        result[f"basisPct{other_name.title()}"] = (other_close - canonical_close) / canonical_close

        other_ret = other_close.pct_change()
        result[f"{other_name}Ret"] = other_ret
        result[f"retDivergence{other_name.title()}"] = other_ret - canonical_ret

        for lag in [1, 2, 3, 5]:
            result[f"{other_name}RetLag{lag}"] = other_ret.shift(lag)

        other_vol_col = f"{other_name}Volume"
        canon_vol_col = f"{canonical_name}Volume"
        if other_vol_col in df.columns and canon_vol_col in df.columns:
            result[f"volRatio{other_name.title()}"] = (
                df[other_vol_col] / df[canon_vol_col].replace(0, np.nan)
            )

    result["realizedVol"] = canonical_ret.rolling(20).std()
    result["realizedVol5"] = canonical_ret.rolling(5).std()

    return result


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from candle data."""
    if _is_multi_exchange(df):
        return _compute_multi_exchange_features(df)
    return _compute_single_exchange_features(df)
