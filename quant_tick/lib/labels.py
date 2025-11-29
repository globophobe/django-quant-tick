import hashlib
import logging

import numpy as np
from pandas import DataFrame

from quant_tick.lib.ml import (
    DEFAULT_ASYMMETRIES,
    DEFAULT_WIDTHS,
    generate_multi_config_labels,
)
from quant_tick.models import Candle, MLConfig, MLFeatureData, Symbol

logger = logging.getLogger(__name__)


def generate_labels_from_config(config: MLConfig) -> MLFeatureData | None:
    """Generate labels from config.

    Args:
        config: MLConfig instance with candle, symbol, json_data

    Returns:
        MLFeatureData if successful, None otherwise
    """
    candle = config.candle
    decision_horizons = config.json_data.get("decision_horizons", [60, 120, 180])
    widths = config.json_data.get("widths", DEFAULT_WIDTHS)
    asymmetries = config.json_data.get("asymmetries", DEFAULT_ASYMMETRIES)
    min_bars = 1000

    # Load candle data
    df = candle.get_candle_data()
    if df is None or len(df) < min_bars:
        n = len(df) if df is not None else 0
        logger.error(f"{config}: insufficient data ({n}/{min_bars})")
        return None

    logger.info(f"{config}: loaded {len(df)} bars with {len(df.columns)} columns")

    # Compute derived features from per-exchange columns
    df = _compute_features(df)

    # Generate labels
    labeled = generate_multi_config_labels(
        df,
        widths=widths,
        asymmetries=asymmetries,
        decision_horizons=decision_horizons,
    )

    n_configs = len(widths) * len(asymmetries)
    logger.info(
        f"{config}: generated {len(labeled)} rows "
        f"({len(df)} bars x {n_configs} configs)"
    )

    # Compute schema hash
    schema_cols = sorted(labeled.columns.tolist())
    schema_hash = hashlib.sha256(",".join(schema_cols).encode()).hexdigest()[:16]

    # Save to MLFeatureData (don't modify config)
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

    return feature_data


def generate_labels(
    candle: Candle,
    symbol: Symbol,
    decision_horizons: list[int] | None = None,
    min_bars: int = 1000,
    widths: list[float] | None = None,
    asymmetries: list[float] | None = None,
) -> MLConfig | None:
    """Generate labels.

    Args:
        candle: Candle to generate labels for
        symbol: Symbol for position tracking
        decision_horizons: List of decision horizons in bars
        min_bars: Minimum bars required
        widths: Range widths
        asymmetries: Asymmetries

    Returns:
        MLConfig or None
    """
    decision_horizons = decision_horizons or [60, 120, 180]
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
        decision_horizons=decision_horizons,
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
    max_horizon = max(decision_horizons)
    config, created = MLConfig.objects.get_or_create(
        candle=candle,
        symbol=symbol,
        defaults={
            "horizon_bars": max_horizon,
            "json_data": {
                "decision_horizons": decision_horizons,
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


def _compute_features(df: DataFrame) -> DataFrame:
    """Compute features."""
    if _is_multi_exchange(df):
        return _compute_multi_exchange_features(df)
    else:
        return _compute_single_exchange_features(df)


def _is_multi_exchange(df: DataFrame) -> bool:
    """Is multi exchange?"""
    return any(c.endswith("Close") and c != "close" for c in df.columns)


def _compute_single_exchange_features(data_frame: DataFrame) -> DataFrame:
    """Compute single exchange features."""
    df = data_frame.copy()
    returns = df["close"].pct_change()

    # Volatility features
    df["realizedVol"] = returns.rolling(20).std()
    df["realizedVol5"] = returns.rolling(5).std()

    # Vol regime indicators
    vol_slow = returns.rolling(60).std()
    df["volRatio"] = df["realizedVol5"] / df["realizedVol"]
    df["volZScore"] = (df["realizedVol"] - vol_slow) / vol_slow.rolling(60).std()
    df["volPercentile"] = df["realizedVol"].rolling(100).rank(pct=True)
    df["isHighVol"] = (df["volPercentile"] > 0.75).astype(int)
    df["isLowVol"] = (df["volPercentile"] < 0.25).astype(int)

    # Rolling Sharpe ratio (momentum quality)
    rolling_mean = returns.rolling(20).mean()
    rolling_std = returns.rolling(20).std()
    df["rollingSharpe20"] = rolling_mean / (rolling_std + 1e-8)

    return df


def _compute_multi_exchange_features(data_frame: DataFrame) -> DataFrame:
    """Compute multi exchange features."""
    df = data_frame.copy()

    close_cols = [c for c in data_frame.columns if c.endswith("Close") and c != "close"]
    canonical_col = close_cols[0]
    canonical_name = canonical_col.replace("Close", "")
    canonical_close = data_frame[canonical_col]

    if "close" not in df.columns:
        df["close"] = canonical_close

    canonical_returns = canonical_close.pct_change()
    df[f"{canonical_name}Ret"] = canonical_returns

    for close_col in close_cols[1:]:
        other_name = close_col.replace("Close", "")
        other_close = df[close_col]

        df[f"{other_name}Missing"] = other_close.isna().astype(int)
        df[f"basis{other_name.title()}"] = other_close - canonical_close
        df[f"basisPct{other_name.title()}"] = (other_close - canonical_close) / canonical_close

        other_returns = other_close.pct_change()
        df[f"{other_name}Ret"] = other_returns
        df[f"retDivergence{other_name.title()}"] = other_returns - canonical_returns

        for lag in [1, 2, 3, 5]:
            df[f"{other_name}RetLag{lag}"] = other_returns.shift(lag)

        other_vol_col = f"{other_name}Volume"
        canon_vol_col = f"{canonical_name}Volume"
        if other_vol_col in df.columns and canon_vol_col in df.columns:
            df[f"volRatio{other_name.title()}"] = (
                df[other_vol_col] / df[canon_vol_col].replace(0, np.nan)
            )

    # Volatility features
    df["realizedVol"] = canonical_returns.rolling(20).std()
    df["realizedVol5"] = canonical_returns.rolling(5).std()

    # Vol regime indicators
    vol_slow = canonical_returns.rolling(60).std()
    df["volRatio"] = df["realizedVol5"] / df["realizedVol"]
    df["volZScore"] = (df["realizedVol"] - vol_slow) / vol_slow.rolling(60).std()
    df["volPercentile"] = df["realizedVol"].rolling(100).rank(pct=True)
    df["isHighVol"] = (df["volPercentile"] > 0.75).astype(int)
    df["isLowVol"] = (df["volPercentile"] < 0.25).astype(int)

    # Rolling Sharpe ratio
    rolling_mean = canonical_returns.rolling(20).mean()
    rolling_std = canonical_returns.rolling(20).std()
    df["rollingSharpe20"] = rolling_mean / (rolling_std + 1e-8)
    return df
