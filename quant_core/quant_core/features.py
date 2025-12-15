"""Feature engineering for ML models."""

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

    if "close" not in df.columns:
        df["close"] = canonical_close

    canonical_returns = canonical_close.pct_change()
    df[f"{canonical_name}Ret"] = canonical_returns

    # Loop over all non-canonical exchanges
    for close_col in [c for c in close_cols if c != canonical_col]:
        other_name = close_col.replace("Close", "")
        other_close = df[close_col]

        df[f"{other_name}Missing"] = other_close.isna().astype(int)
        df[f"basis{other_name.title()}"] = other_close - canonical_close
        df[f"basisPct{other_name.title()}"] = (
            other_close - canonical_close
        ) / canonical_close

        other_returns = other_close.pct_change()
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
    return df
