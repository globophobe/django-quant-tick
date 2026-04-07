import datetime

import pandas as pd
from pandas import DataFrame

from quant_tick.constants import ZERO

from .calendar import iter_once, iter_window
from .dataframe import is_decimal_close


def aggregate_trades(data_frame: DataFrame) -> DataFrame:
    """Aggregate trades

    1) in the same direction, either buy or sell
    2) at the same timestamp, and nanoseconds

    Resulting aggregation was either a single market order, or a
    cascade of executed orders.
    """
    if not len(data_frame):
        return pd.DataFrame([])

    has_symbol = "symbol" in data_frame.columns
    samples = []
    iterator = data_frame.itertuples(index=False)
    first_row = next(iterator)
    uid = first_row.uid
    timestamp = first_row.timestamp
    nanoseconds = first_row.nanoseconds
    tick_rule = first_row.tickRule
    price = first_row.price
    volume = first_row.volume
    notional = first_row.notional
    ticks = 1
    symbol = first_row.symbol if has_symbol else None
    for row in iterator:
        is_same_sample = (
            row.timestamp == timestamp
            and row.nanoseconds == nanoseconds
            and row.tickRule == tick_rule
            and (not has_symbol or row.symbol == symbol)
        )
        if is_same_sample:
            price = row.price
            volume += row.volume
            notional += row.notional
            ticks += 1
            continue
        data = {
            "uid": uid,
            "timestamp": timestamp,
            "nanoseconds": nanoseconds,
            "price": price,
            "volume": volume,
            "notional": notional,
            "ticks": ticks,
            "tickRule": tick_rule,
        }
        if has_symbol:
            data["symbol"] = symbol
        samples.append(data)
        uid = row.uid
        timestamp = row.timestamp
        nanoseconds = row.nanoseconds
        tick_rule = row.tickRule
        price = row.price
        volume = row.volume
        notional = row.notional
        ticks = 1
        if has_symbol:
            symbol = row.symbol
    data = {
        "uid": uid,
        "timestamp": timestamp,
        "nanoseconds": nanoseconds,
        "price": price,
        "volume": volume,
        "notional": notional,
        "ticks": ticks,
        "tickRule": tick_rule,
    }
    if has_symbol:
        data["symbol"] = symbol
    samples.append(data)
    # Assert volume equal.
    aggregated = pd.DataFrame(samples)
    is_close = is_decimal_close(data_frame.volume.sum(), aggregated.volume.sum())
    assert is_close, "Volume is not equal."
    return aggregated


def filter_by_timestamp(
    data_frame: DataFrame,
    timestamp_from: datetime,
    timestamp_to: datetime,
    inclusive: bool = False,
) -> DataFrame:
    """Filter by timestamp."""
    if len(data_frame):
        attr = (
            "index" if isinstance(data_frame.index, pd.DatetimeIndex) else "timestamp"
        )
        a = getattr(data_frame, attr)
        lower_bound = a >= timestamp_from
        if inclusive:
            upper_bound = a <= timestamp_to
        else:
            upper_bound = a < timestamp_to
        return data_frame[lower_bound & upper_bound]
    else:
        return pd.DataFrame([])


def volume_filter_with_time_window(
    data_frame: DataFrame, min_volume: int = 1000, window: str = "1min"
) -> DataFrame:
    """Volume filter, with time window."""
    samples = []
    if len(data_frame):
        timestamp_from = data_frame.iloc[0].timestamp
        # Iterator is not inclusive of timestamp_to, so increase by 1.
        timestamp_to = data_frame.iloc[-1].timestamp + pd.Timedelta("1min")
        if window:
            # Chunk data_frame by window.
            iterator = iter_window(timestamp_from, timestamp_to, window)
        else:
            iterator = iter_once(timestamp_from, timestamp_to)
        for ts_from, ts_to in iterator:
            df = filter_by_timestamp(data_frame, ts_from, ts_to)
            if len(df):
                start = 0
                df = df.reset_index(drop=True)
                for index, volume in enumerate(df.volume):
                    is_min_volume = volume >= min_volume if min_volume else True
                    if is_min_volume:
                        sample = df.iloc[start : index + 1]
                        samples.append(
                            volume_filter(sample, is_min_volume=is_min_volume)
                        )
                        start = index + 1
                if start < len(df):
                    sample = df.iloc[start:]
                    samples.append(volume_filter(sample))
    return pd.DataFrame(samples)


def volume_filter(df: DataFrame, is_min_volume: bool = False) -> dict:
    """Volume filter."""
    last_row = df.iloc[-1]
    data = {
        "uid": last_row.uid,
        "timestamp": last_row.timestamp,
        "nanoseconds": last_row.nanoseconds,
    }
    if is_min_volume:
        data.update(
            {
                "price": last_row.price,
                "volume": last_row.volume,
                "notional": last_row.notional,
                "tickRule": last_row.tickRule,
                "ticks": last_row.ticks,
            }
        )
    else:
        data.update(
            {
                "price": last_row.price,
                "volume": None,
                "notional": None,
                "tickRule": None,
                "ticks": None,
            }
        )
    has_totals = "totalVolume" in df.columns
    if has_totals:
        data.update(
            {
                "high": df.high.max(),
                "low": df.low.min(),
                "totalBuyVolume": df.totalBuyVolume.sum() or ZERO,
                "totalVolume": df.totalVolume.sum() or ZERO,
                "totalBuyNotional": df.totalBuyNotional.sum() or ZERO,
                "totalNotional": df.totalNotional.sum() or ZERO,
                "totalBuyTicks": int(df.totalBuyTicks.sum()),
                "totalTicks": int(df.totalTicks.sum()),
            }
        )
    else:
        is_buy = df.tickRule == 1
        data.update(
            {
                "high": df.price.max(),
                "low": df.price.min(),
                "totalBuyVolume": df.loc[is_buy, "volume"].sum() or ZERO,
                "totalVolume": df.volume.sum() or ZERO,
                "totalBuyNotional": df.loc[is_buy, "notional"].sum() or ZERO,
                "totalNotional": df.notional.sum() or ZERO,
                "totalBuyTicks": df.loc[is_buy, "ticks"].sum(),
                "totalTicks": df.ticks.sum(),
            }
        )
    return data
