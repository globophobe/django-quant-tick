from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
from pandas import DataFrame

from .aggregate import filter_by_timestamp
from .calendar import iter_window
from .dataframe import is_decimal_close

ZERO = Decimal("0")


def candles_to_data_frame(
    timestamp_from: datetime,
    timestamp_to: datetime,
    candles: list[dict],
    reverse: bool = True,
) -> DataFrame:
    """Get candle data_frame."""
    data_frame = pd.DataFrame(candles)
    df = filter_by_timestamp(
        data_frame,
        timestamp_from,
        timestamp_to,
        inclusive=timestamp_from == timestamp_to,
    )
    if len(df):
        df.set_index("timestamp", inplace=True)
    # REST API, data is reverse order.
    return df.iloc[::-1] if reverse else df


def resample_candles(
    data_frame: DataFrame,
    *,
    timestamp_from: datetime,
    timestamp_to: datetime,
    resolution_minutes: int,
) -> DataFrame:
    """Aggregate candle to a coarser resolution."""
    if data_frame.empty:
        return data_frame
    frame = data_frame.reset_index().copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    )
    frame["bucket"] = frame["timestamp"].dt.floor(f"{resolution_minutes}min")
    aggregations = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    for column in ("volume", "notional"):
        if column in frame.columns:
            aggregations[column] = "sum"
    grouped = frame.groupby("bucket", sort=True).agg(aggregations).reset_index()
    grouped = grouped.rename(columns={"bucket": "timestamp"})
    return candles_to_data_frame(
        timestamp_from,
        timestamp_to,
        grouped.to_dict("records"),
        reverse=False,
    )


def aggregate_candles(
    data_frame: DataFrame,
    timestamp_from: datetime,
    timestamp_to: datetime,
    window: str = "1min",
    as_data_frame: bool = True,
    min_volume_exponent: int | None = None,
    min_notional_exponent: int | None = None,
) -> list[dict]:
    """Aggregate candles"""
    data = []
    for ts_from, ts_to in iter_window(timestamp_from, timestamp_to, window):
        df = filter_by_timestamp(data_frame, ts_from, ts_to)
        if len(df):
            candle = aggregate_candle(
                df,
                timestamp=ts_from,
                min_volume_exponent=min_volume_exponent,
                min_notional_exponent=min_notional_exponent,
            )
            data.append(candle)
        else:
            d = {"timestamp": ts_from}
            for k in "open", "high", "low", "close":
                d[k] = None
            for k in (
                "volume",
                "buyVolume",
                "notional",
                "buyNotional",
                "ticks",
                "buyTicks",
            ):
                d[k] = ZERO
    if as_data_frame:
        df = DataFrame(data)
        if len(df):
            df.set_index("timestamp", inplace=True)
        return df
    else:
        return data


def aggregate_candle(
    data_frame: DataFrame,
    timestamp: datetime | None = None,
    min_volume_exponent: int | None = None,
    min_notional_exponent: int | None = None,
) -> dict:
    """Aggregate candle."""
    ts = timestamp if timestamp else data_frame.iloc[0].timestamp
    data = {"timestamp": ts}
    data.update(_aggregate_ohlc(data_frame))
    data.update(
        _aggregate_totals(
            data_frame,
            min_volume_exponent,
            min_notional_exponent,
        )
    )
    data.update(_aggregate_realized_variance(data_frame))
    return data


def _aggregate_ohlc(df: DataFrame) -> dict:
    """Aggregate OHLC from trade data."""
    if "price" in df.columns:
        return {
            "open": df.iloc[0].price,
            "high": df.price.max(),
            "low": df.price.min(),
            "close": df.iloc[-1].price,
        }
    return {
        "open": df.iloc[0].open,
        "high": df.high.max(),
        "low": df.low.min(),
        "close": df.iloc[-1].close,
    }


def _aggregate_totals(
    df: DataFrame,
    min_volume_exponent: int | None = None,
    min_notional_exponent: int | None = None,
) -> dict:
    """Aggregate volume, notional, ticks, and round fields."""
    data = {}
    has_totals = "totalVolume" in df.columns

    if has_totals:
        data["volume"] = df.totalVolume.sum()
        data["buyVolume"] = df.totalBuyVolume.sum()
        data["notional"] = df.totalNotional.sum()
        data["buyNotional"] = df.totalBuyNotional.sum()
        data["ticks"] = int(df.totalTicks.sum())
        data["buyTicks"] = int(df.totalBuyTicks.sum())
    else:
        is_buy = df.tickRule == 1
        data["volume"] = df.volume.sum()
        data["buyVolume"] = df.loc[is_buy, "volume"].sum()
        data["notional"] = df.notional.sum()
        data["buyNotional"] = df.loc[is_buy, "notional"].sum()
        if "ticks" in df.columns:
            data["ticks"] = int(df.ticks.sum())
            data["buyTicks"] = int(df.loc[is_buy, "ticks"].sum())
        else:
            data["ticks"] = len(df)
            data["buyTicks"] = int(is_buy.sum())
    if min_volume_exponent is not None:
        data["roundVolume"] = ZERO
        data["roundBuyVolume"] = ZERO
        data["roundVolumeSumNotional"] = ZERO
        data["roundBuyVolumeSumNotional"] = ZERO
    if min_notional_exponent is not None:
        data["roundNotional"] = ZERO
        data["roundBuyNotional"] = ZERO
        data["roundNotionalSumVolume"] = ZERO
        data["roundBuyNotionalSumVolume"] = ZERO
    if min_volume_exponent is not None or min_notional_exponent is not None:
        for row in df.itertuples(index=False):
            if min_volume_exponent is not None and is_round_volume(
                row.volume, min_volume_exponent
            ):
                data["roundVolume"] += row.volume
                data["roundVolumeSumNotional"] += row.notional
                if row.tickRule == 1:
                    data["roundBuyVolume"] += row.volume
                    data["roundBuyVolumeSumNotional"] += row.notional
            if min_notional_exponent is not None and is_round_notional(
                row.notional, min_notional_exponent
            ):
                data["roundNotional"] += row.notional
                data["roundNotionalSumVolume"] += row.volume
                if row.tickRule == 1:
                    data["roundBuyNotional"] += row.notional
                    data["roundBuyNotionalSumVolume"] += row.volume
    return data


def is_round_volume(
    volume: int | Decimal, min_volume_exponent: int, base_unit: int = 1000
) -> bool:
    """Whether volume matches the configured round-dollar threshold."""
    if not volume or volume % 1 != 0:
        return False
    required_unit = base_unit * (10 ** (min_volume_exponent - 1))
    return int(volume) % required_unit == 0


def is_round_notional(
    notional: Decimal, min_notional_exponent: int
) -> bool:
    """Whether notional matches the configured round-size threshold."""
    if not notional:
        return False
    required_unit = Decimal("0.1") * (Decimal("10") ** (min_notional_exponent - 1))
    return notional % required_unit == 0


def _aggregate_realized_variance(df: DataFrame) -> dict:
    """Compute realized variance."""
    if len(df) == 0:
        return {"realizedVariance": ZERO}

    price_col = "price" if "price" in df.columns else "close"
    if price_col not in df.columns:
        return {"realizedVariance": ZERO}

    if len(df) > 1:
        prices = df[price_col].astype(float)
        prices = prices[prices > 0]
        if len(prices) <= 1:
            return {"realizedVariance": ZERO}
        log_prices = np.log(prices)
        log_returns = log_prices.diff().dropna()
        return {"realizedVariance": Decimal(str((log_returns**2).sum()))}

    return {"realizedVariance": ZERO}


def validate_aggregated_candles(
    aggregated_candles: DataFrame, exchange_candles: DataFrame
) -> bool | None:
    """Validate data_frame with candles from Exchange API."""
    if not len(exchange_candles):
        return None

    if "notional" in exchange_candles.columns:
        key = "notional"
    elif "volume" in exchange_candles.columns:
        key = "volume"
    else:
        raise NotImplementedError

    ok = None
    if len(aggregated_candles):
        missing_exchange_rows = False
        has_false = False
        for row in aggregated_candles.itertuples():
            timestamp = row.Index
            try:
                candle = exchange_candles.loc[timestamp]
            # Candle may be missing from API result.
            except KeyError:
                missing_exchange_rows = True
            else:
                value = getattr(candle, key)
                if isinstance(value, int):
                    value = Decimal(value)
                if not is_decimal_close(getattr(row, key), value):
                    has_false = True
        if has_false:
            ok = False
        elif missing_exchange_rows:
            ok = None
        else:
            ok = True
    elif exchange_candles[key].sum() == 0:
        ok = True

    if ok in (True, None):
        # Maybe candle with no volume or notional.
        missing = exchange_candles.index.difference(aggregated_candles.index)
        if len(missing):
            # If missing candles have volume or notional, then ok should be False.
            # Maybe validates on retry.
            if exchange_candles.loc[missing, key].sum() != 0:
                ok = False

    return ok
