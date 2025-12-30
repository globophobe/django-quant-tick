"""Feature engineering utilities for ML models."""

import numpy as np
from pandas import DataFrame


def _compute_features(
    df: DataFrame, canonical_exchange: str | None = None
) -> DataFrame:
    """Compute features.

    Args:
        df: Candle data
        canonical_exchange: Exchange to use as canonical for multi-exchange (e.g., "coinbase")
                           Required for multi-exchange candles, raises ValueError if missing.
                           Ignored for single-exchange candles.
    """
    if _is_multi_exchange(df):
        return _compute_multi_exchange_features(df, canonical_exchange)
    return _compute_single_exchange_features(df)


def _is_multi_exchange(df: DataFrame) -> bool:
    """Is multi exchange?"""
    return any(c.endswith("Close") and c != "close" for c in df.columns)


def _compute_single_exchange_features(data_frame: DataFrame) -> DataFrame:
    """Compute single exchange features."""
    df = data_frame.copy()
    returns = df["close"].pct_change(fill_method=None)

    # To prevent train/serve skew
    max_warmup = compute_max_warmup_bars()
    df["has_full_warmup"] = (np.arange(len(df)) >= max_warmup).astype(int)

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


def _compute_multi_exchange_features(
    data_frame: DataFrame, canonical_exchange: str | None
) -> DataFrame:
    """Compute multi exchange features.

    Args:
        data_frame: Candle data with per-exchange columns
        canonical_exchange: Exchange to use as canonical (e.g., "coinbase").
                           Must be provided, raises ValueError if None.

    Raises:
        ValueError: If canonical_exchange is None or not found in data
    """
    df = data_frame.copy()

    close_cols = [c for c in data_frame.columns if c.endswith("Close") and c != "close"]

    if not canonical_exchange:
        raise ValueError(
            "canonical_exchange is required for multi-exchange candles. "
            f"Available exchanges: {[c.replace('Close', '') for c in close_cols]}"
        )

    # Use specified canonical exchange
    preferred_col = f"{canonical_exchange}Close"
    if preferred_col not in close_cols:
        raise ValueError(
            f"Canonical exchange '{canonical_exchange}' not found in candle data. "
            f"Available exchanges: {[c.replace('Close', '') for c in close_cols]}"
        )

    canonical_col = preferred_col
    canonical_name = canonical_col.replace("Close", "")
    canonical_close = data_frame[canonical_col]

    # Set canonical OHLC columns
    if "close" not in df.columns:
        df["close"] = canonical_close
    if "low" not in df.columns and f"{canonical_name}Low" in df.columns:
        df["low"] = df[f"{canonical_name}Low"]
    if "high" not in df.columns and f"{canonical_name}High" in df.columns:
        df["high"] = df[f"{canonical_name}High"]
    if "open" not in df.columns and f"{canonical_name}Open" in df.columns:
        df["open"] = df[f"{canonical_name}Open"]

    canonical_returns = canonical_close.pct_change(fill_method=None)
    df[f"{canonical_name}Ret"] = canonical_returns

    # To prevent train/serve skew
    max_warmup = compute_max_warmup_bars()
    df["has_full_warmup"] = (np.arange(len(df)) >= max_warmup).astype(int)

    # Loop over all non-canonical exchanges
    for close_col in [c for c in close_cols if c != canonical_col]:
        other_name = close_col.replace("Close", "")
        other_close = df[close_col]

        df[f"{other_name}Missing"] = other_close.isna().astype(int)
        df[f"basis{other_name.title()}"] = other_close - canonical_close
        df[f"basisPct{other_name.title()}"] = (
            other_close - canonical_close
        ) / canonical_close

        other_returns = other_close.pct_change(fill_method=None)
        df[f"{other_name}Ret"] = other_returns
        df[f"retDivergence{other_name.title()}"] = other_returns - canonical_returns

        for lag in [1, 2, 3, 5]:
            df[f"{other_name}RetLag{lag}"] = other_returns.shift(lag)

        other_vol_col = f"{other_name}Volume"
        canon_vol_col = f"{canonical_name}Volume"
        if other_vol_col in df.columns and canon_vol_col in df.columns:
            df[f"volRatio{other_name.title()}"] = df[other_vol_col] / df[
                canon_vol_col
            ].replace(0, np.nan)

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

    # OFI features (Order Flow Imbalance) per exchange
    exchanges = [c.replace("Close", "") for c in close_cols]
    ofi_series = {}
    notional_series = {}

    for exchange in exchanges:
        buy_vol_col = f"{exchange}BuyVolume"
        vol_col = f"{exchange}Volume"
        notional_col = f"{exchange}Notional"

        if buy_vol_col in df.columns and vol_col in df.columns:
            # OFI = buyVolume / totalVolume - 0.5 (ranges from -0.5 to 0.5)
            ofi = df[buy_vol_col] / df[vol_col].replace(0, np.nan) - 0.5
            df[f"{exchange}Ofi"] = ofi
            df[f"{exchange}OfiMa5"] = ofi.rolling(5).mean()
            df[f"{exchange}OfiMa20"] = ofi.rolling(20).mean()
            ofi_series[exchange] = ofi

        if notional_col in df.columns:
            notional_series[exchange] = df[notional_col]

    # Aggregate OFI weighted by notional
    if ofi_series and notional_series:
        common_exchanges = set(ofi_series.keys()) & set(notional_series.keys())
        if common_exchanges:
            total_notional = sum(notional_series[ex] for ex in common_exchanges)
            weighted_ofi = sum(
                ofi_series[ex] * notional_series[ex] / total_notional.replace(0, np.nan)
                for ex in common_exchanges
            )
            df["aggregateOfi"] = weighted_ofi
            df["aggregateOfiMa5"] = weighted_ofi.rolling(5).mean()
            df["aggregateOfiMa20"] = weighted_ofi.rolling(20).mean()

    return df


def compute_max_warmup_bars() -> int:
    """Compute maximum warmup bars required for feature computation.

    Returns minimum number of bars needed for all features to be fully warmed up.
    Based on rolling window sizes in _compute_single_exchange_features and
    _compute_multi_exchange_features.

    Returns:
        Maximum warmup bars across all features
    """
    warmup_requirements = {
        "realizedVol": 20,  # rolling(20).std()
        "realizedVol5": 5,  # rolling(5).std()
        "volRatio": 20,  # max(realizedVol5, realizedVol)
        "volZScore": 120,  # vol_slow.rolling(60).std() needs 60 + vol_slow needs 60
        "volPercentile": 100,  # rolling(100).rank()
        "isHighVol": 100,  # depends on volPercentile
        "isLowVol": 100,  # depends on volPercentile
        "rollingSharpe20": 20,  # rolling(20) for mean and std
        "ofi_ma5": 5,  # rolling(5).mean()
        "ofi_ma20": 20,  # rolling(20).mean()
    }
    return max(warmup_requirements.values())
