from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
from pandas import DataFrame

from .aggregate import aggregate_sum, filter_by_timestamp
from .calendar import get_range, iter_window
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


def aggregate_candles(
    data_frame: DataFrame,
    timestamp_from: datetime,
    timestamp_to: datetime,
    window: str = "1t",
    as_data_frame: bool = True,
) -> list[dict]:
    """Aggregate candles"""
    data = []
    for ts_from, ts_to in iter_window(timestamp_from, timestamp_to, window):
        df = filter_by_timestamp(data_frame, ts_from, ts_to)
        if len(df):
            candle = aggregate_candle(df, timestamp=ts_from)
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
    return DataFrame(data) if as_data_frame else data


def aggregate_candle(data_frame: DataFrame, timestamp: datetime | None = None) -> dict:
    """Aggregate candle"""
    first_row = data_frame.iloc[0]
    last_row = data_frame.iloc[-1]
    high = data_frame.price.max()
    low = data_frame.price.min()
    buy_data_frame = data_frame[data_frame.tickRule == 1]
    if "totalVolume" in data_frame.columns:
        volume = data_frame.totalVolume.sum()
    else:
        volume = data_frame.volume.sum()
    if "totalBuyVolume" in data_frame.columns:
        buy_volume = data_frame.totalBuyVolume.sum()
    else:
        buy_volume = buy_data_frame.volume.sum()
    if "totalNotional" in data_frame.columns:
        notional = data_frame.totalNotional.sum()
    else:
        notional = data_frame.notional.sum()
    if "totalBuyNotional" in data_frame.columns:
        buy_notional = data_frame.totalBuyNotional.sum()
    else:
        buy_notional = buy_data_frame.notional.sum()
    if "totalTicks" in data_frame.columns:
        ticks = int(data_frame.totalTicks.sum())
    else:
        ticks = int(data_frame.ticks.sum())
    if "totalBuyTicks" in data_frame.columns:
        buy_ticks = int(data_frame.totalBuyTicks.sum())
    else:
        buy_ticks = int(buy_data_frame.ticks.sum())
    return {
        "timestamp": timestamp if timestamp else first_row.timestamp,
        "open": first_row.price,
        "high": high,
        "low": low,
        "close": last_row.price,
        "volume": volume,
        "buyVolume": buy_volume,
        "notional": notional,
        "buyNotional": buy_notional,
        "ticks": ticks,
        "buyTicks": buy_ticks,
    }


def validate_data_frame(
    timestamp_from: datetime,
    timestamp_to: datetime,
    data_frame: DataFrame,
    candles: DataFrame,
) -> dict:
    """Validate data_frame with candles from Exchange API."""
    if len(candles):
        if "notional" in candles.columns:
            key = "notional"
        elif "volume" in candles.columns:
            key = "volume"
        else:
            raise NotImplementedError
        validated = {
            candle.Index: True
            if (value := getattr(candle, key)) == Decimal("0")
            else value
            for candle in candles.itertuples()
        }
        k = key.title()
        capitalized_key = key.capitalize()
        total_key = f"total{capitalized_key}"
        if len(data_frame):
            # If there was a significant trade filter, total_key
            attrs = total_key if total_key in data_frame.columns else key
            df = aggregate_sum(data_frame, attrs=attrs, window="1t")
            for row in df.itertuples():
                timestamp = row.Index
                try:
                    candle = candles.loc[timestamp]
                # Candle may be missing from API result.
                except KeyError:
                    validated[timestamp] = None
                else:
                    values = row[1], candle[key]
                    is_close = is_decimal_close(*values)
                    if is_close:
                        validated[timestamp] = True
                    else:
                        # Maybe int64
                        if isinstance(candle[key], np.int64):
                            v = int(candle[key])
                        else:
                            v = candle[key]
                        validated[timestamp] = {key: row[1], f"exchange{k}": v}
        # Candle and trade API data availability may differ.
        for timestamp, v in validated.items():
            if isinstance(v, Decimal):
                validated[timestamp] = {key: Decimal("0"), f"exchange{k}": v}
    else:
        validated = {
            timestamp: None
            for timestamp in get_range(timestamp_from, timestamp_to)
            if timestamp >= timestamp_from and timestamp < timestamp_to
        }
    return validated
