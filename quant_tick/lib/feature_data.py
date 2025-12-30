import logging
from datetime import datetime
from decimal import Decimal

import pandas as pd
from quant_core.features import _compute_features

from quant_tick.lib.ml import generate_labels
from quant_tick.models import CandleData, MLConfig

logger = logging.getLogger(__name__)


def _flatten_exchange_data(json_data: dict) -> dict:
    """Flatten nested exchange data to columns like binanceClose, coinbaseVolume, etc."""
    if "exchanges" not in json_data:
        return json_data  # Already flat or single-exchange

    flat = {}
    for exchange, data in json_data["exchanges"].items():
        for key, value in data.items():
            # Capitalize first letter: close -> Close, buyVolume -> BuyVolume
            col_name = f"{exchange}{key[0].upper()}{key[1:]}"
            flat[col_name] = value
    return flat


def load_training_df(
    config: MLConfig,
    timestamp_from: datetime | None = None,
    timestamp_to: datetime | None = None,
) -> pd.DataFrame:
    """Load training DataFrame with features computed on-the-fly.

    Drops initial warmup rows (data hygiene) to remove low-information samples
    where many features have NaN/sentinel values due to insufficient history.

    1. Load all CandleData for config.candle in the date range
    2. Compute features from OHLCV data
    3. Drop first max_warmup_bars rows (removes high-NaN period)
    4. Generate labels using generate_labels()

    Args:
        config: MLConfig instance
        timestamp_from: Start of training period (auto-detected if None)
        timestamp_to: End of training period (auto-detected if None)

    Returns:
        DataFrame with expanded grid (n_bars × n_configs rows)
        where n_bars excludes initial warmup rows

    Raises:
        ValueError: If no CandleData found or insufficient data after warmup
    """
    # Auto-detect date range if not provided
    if timestamp_from is None or timestamp_to is None:
        candle_bounds = CandleData.objects.filter(candle=config.candle)
        if not candle_bounds.exists():
            raise ValueError(f"{config}: No CandleData found")

        if timestamp_from is None:
            timestamp_from = candle_bounds.order_by("timestamp").first().timestamp
        if timestamp_to is None:
            last = candle_bounds.order_by("-timestamp").first()
            timestamp_to = last.timestamp + pd.Timedelta("1h")

    candle_data_qs = CandleData.objects.filter(
        candle=config.candle,
        timestamp__gte=timestamp_from,
        timestamp__lt=timestamp_to,
    ).order_by("timestamp")

    if not candle_data_qs.exists():
        msg = f"{config}: No CandleData in range {timestamp_from} to {timestamp_to}"
        raise ValueError(msg)

    # Load using values_list for efficiency
    import time

    logger.info(f"Loading candles from {timestamp_from} to {timestamp_to}")

    load_start = time.time()
    candle_data = list(candle_data_qs.values_list("timestamp", "json_data"))
    load_elapsed = time.time() - load_start

    logger.info(f"Loaded {len(candle_data)} candles in {load_elapsed:.1f}s")

    # Build DataFrame with flattened OHLCV
    rows = [
        {"timestamp": ts, **_flatten_exchange_data(jd)} for ts, jd in candle_data
    ]
    df = pd.DataFrame(rows)

    # Convert Decimal columns to float, skip datetime/timestamp objects
    for col in df.columns:
        if df[col].dtype == object:
            # Only convert if it's actually numeric-like (Decimal, numeric strings)
            # Skip datetime/timestamp objects to avoid silently creating all-NaN
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if sample is not None and isinstance(sample, (Decimal, int, float)):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            # Else: leave as-is (will be filtered out later by select_dtypes in ml.py)

    logger.info(f"{config}: Loaded {len(df)} candles, computing features...")

    # Compute features on-the-fly
    canonical_exchange = config.symbol.exchange
    df = _compute_features(df, canonical_exchange=canonical_exchange)

    # CRITICAL: Drop initial warmup rows (data hygiene)
    from quant_core.features import compute_max_warmup_bars

    max_warmup = compute_max_warmup_bars()
    if len(df) <= max_warmup:
        raise ValueError(
            f"{config}: Insufficient data for training. "
            f"Have {len(df)} bars, need > {max_warmup} bars "
            f"({max_warmup} warmup + data for training/test splits)"
        )

    df_warmed = df.iloc[max_warmup:].reset_index(drop=True)
    n_dropped = len(df) - len(df_warmed)

    logger.info(
        f"{config}: Dropped first {n_dropped} bars (warmup period), "
        f"using {len(df_warmed)} bars for training"
    )

    # Generate labels using multi-horizon approach
    widths = config.get_widths()
    asymmetries = config.get_asymmetries()
    horizons = config.get_horizons()  # Get horizons from config

    logger.info(f"{config}: Generating labels for horizons {horizons}")

    # No batching needed - dataset is bars × configs (not expanded by horizon)
    # Much smaller: ~226k rows vs 41M for hazard approach
    df_labeled = generate_labels(df_warmed, widths, asymmetries, horizons)

    return df_labeled


def load_feature_bars_df(
    config: MLConfig,
    timestamp_from: datetime | None = None,
    timestamp_to: datetime | None = None,
) -> pd.DataFrame:
    """Load feature bars without labels or config expansion.

    Reuses the same pipeline as load_training_df() but stops after:
    1. Loading CandleData with json_data flattening
    2. Computing features via _compute_features()
    3. Dropping warmup rows

    Does NOT:
    - Generate touch labels (generate_labels)
    - Expand to config grid (width/asymmetry combinations)

    Returns:
        DataFrame with (n_bars,) rows
        Columns: timestamp, close, ret, vol, features...
        NO label_* or config columns
    """
    from quant_core.features import compute_max_warmup_bars

    if timestamp_from is None or timestamp_to is None:
        candle_bounds = CandleData.objects.filter(candle=config.candle)
        if not candle_bounds.exists():
            raise ValueError(f"{config}: No CandleData found")

        if timestamp_from is None:
            timestamp_from = candle_bounds.order_by("timestamp").first().timestamp
        if timestamp_to is None:
            last = candle_bounds.order_by("-timestamp").first()
            timestamp_to = last.timestamp + pd.Timedelta("1h")

    candle_data_qs = CandleData.objects.filter(
        candle=config.candle,
        timestamp__gte=timestamp_from,
        timestamp__lt=timestamp_to,
    ).order_by("timestamp")

    if not candle_data_qs.exists():
        msg = f"{config}: No CandleData in range {timestamp_from} to {timestamp_to}"
        raise ValueError(msg)

    import time

    logger.info(f"Loading candles from {timestamp_from} to {timestamp_to}")

    load_start = time.time()
    candle_data = list(candle_data_qs.values_list("timestamp", "json_data"))
    load_elapsed = time.time() - load_start

    logger.info(f"Loaded {len(candle_data)} candles in {load_elapsed:.1f}s")

    rows = [
        {"timestamp": ts, **_flatten_exchange_data(jd)} for ts, jd in candle_data
    ]
    df = pd.DataFrame(rows)

    # Drop per-exchange timestamp columns (datetime64 not supported by LightGBM)
    exchange_ts_cols = [c for c in df.columns if c != "timestamp" and c.endswith("Timestamp")]
    if exchange_ts_cols:
        df = df.drop(columns=exchange_ts_cols)
        logger.info(f"Dropped {len(exchange_ts_cols)} exchange timestamp columns")

    # Coerce all non-timestamp columns to numeric (handles Decimal, numeric strings, mixed types)
    non_ts_cols = [c for c in df.columns if c != "timestamp"]
    df[non_ts_cols] = df[non_ts_cols].apply(pd.to_numeric, errors="coerce")

    logger.info(f"{config}: Loaded {len(df)} candles, computing features...")

    canonical_exchange = config.symbol.exchange
    df = _compute_features(df, canonical_exchange=canonical_exchange)

    # CRITICAL: Add ret column for directional label generation
    # _compute_features() doesn't create this column (only uses returns internally)
    df["ret"] = df["close"].pct_change(fill_method=None)

    # Ensure no object columns remain after feature computation
    for col in df.columns:
        if col != "timestamp" and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            logger.warning(f"Converted straggler column '{col}' from object to numeric")

    max_warmup = compute_max_warmup_bars()
    if len(df) <= max_warmup:
        raise ValueError(
            f"{config}: Insufficient data. "
            f"Have {len(df)} bars, need > {max_warmup} bars"
        )

    df_warmed = df.iloc[max_warmup:].reset_index(drop=True)
    n_dropped = len(df) - len(df_warmed)

    logger.info(
        f"{config}: Dropped first {n_dropped} bars (warmup period), "
        f"using {len(df_warmed)} bars"
    )

    return df_warmed
