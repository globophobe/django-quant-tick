import datetime
import math
from typing import Any

import pandas as pd
from pandas import DataFrame

from quant_tick.constants import ZERO

from .calendar import iter_once, iter_window, to_pydatetime
from .dataframe import is_decimal_close


def is_sample(data_frame: DataFrame, first_index: int, last_index: int) -> bool:
    """Is the range a sample? Short-circuit logic for speed."""
    first_row = data_frame.loc[first_index]
    last_row = data_frame.loc[last_index]
    # For speed, short-circuit
    if first_row.timestamp == last_row.timestamp:
        if first_row.nanoseconds == last_row.nanoseconds:
            if first_row.tickRule == last_row.tickRule:
                if "symbol" in data_frame.columns:
                    if first_row.symbol == last_row.symbol:
                        return False
                else:
                    return False
    return True


def aggregate_trades(data_frame: DataFrame) -> DataFrame:
    """Aggregate trades

    1) in the same direction, either buy or sell
    2) at the same timestamp, and nanoseconds

    Resulting aggregation was either a single market order, or a
    cascade of executed orders.
    """
    df = data_frame.reset_index()
    idx = 0
    samples = []
    total_rows = len(df) - 1
    # Were there two or more trades?
    if len(df) > 1:
        for row in df.itertuples():
            index = row.Index
            last_index = index - 1
            if index > 0:
                is_last_iteration = index == total_rows
                # Is this the last iteration?
                if is_last_iteration:
                    # If equal, one sample
                    if not is_sample(df, idx, index):
                        # Aggregate from idx to end of data frame.
                        sample = df.loc[idx:]
                        samples.append(agg_trades(sample))
                    # Otherwise, two samples.
                    else:
                        # Aggregate from idx to last_index
                        sample = df.loc[idx:last_index]
                        samples.append(agg_trades(sample))
                        # Append last row.
                        sample = df.loc[index:]
                        assert len(sample) == 1
                        samples.append(agg_trades(sample))
                # Is the last row equal to the current row?
                elif is_sample(df, last_index, index):
                    # Aggregate from idx to last_index.
                    sample = df.loc[idx:last_index]
                    aggregated_sample = agg_trades(sample)
                    samples.append(aggregated_sample)
                    idx = index
    # Only one trade in data_frame.
    elif len(df) == 1:
        aggregated_sample = agg_trades(df)
        samples.append(aggregated_sample)
    # Assert volume equal.
    aggregated = pd.DataFrame(samples)
    is_close = is_decimal_close(data_frame.volume.sum(), aggregated.volume.sum())
    assert is_close, "Volume is not equal."
    return aggregated


def agg_trades(data_frame: DataFrame) -> dict[str, Any]:
    """Aggregate trades."""
    first_row = data_frame.iloc[0]
    last_row = data_frame.iloc[-1]
    timestamp = last_row.timestamp
    last_price = last_row.price
    ticks = len(data_frame)
    # Is there more than 1 trade to aggregate?
    if ticks > 1:
        volume = data_frame.volume.sum()
        notional = data_frame.notional.sum()
    else:
        volume = last_row.volume
        notional = last_row.notional
    data = {
        "uid": first_row.uid,
        "timestamp": timestamp,
        "nanoseconds": last_row.nanoseconds,
        "price": last_price,
        "volume": volume,
        "notional": notional,
        "ticks": ticks,
        "tickRule": last_row.tickRule,
    }
    if "symbol" in data_frame.columns:
        data.update({"symbol": last_row.symbol})
    return data


def filter_by_timestamp(
    data_frame: DataFrame,
    timestamp_from: datetime,
    timestamp_to: datetime,
    inclusive: bool = False,
) -> DataFrame:
    """Filter by timestamp."""
    if len(data_frame):
        lower_bound = data_frame.timestamp >= timestamp_from
        if inclusive:
            upper_bound = data_frame.timestamp <= timestamp_to
        else:
            upper_bound = data_frame.timestamp < timestamp_to
        return data_frame[lower_bound & upper_bound]
    else:
        return pd.DataFrame([])


def volume_filter_with_time_window(
    data_frame: DataFrame, min_volume: int = 1000, window: str = "1t"
) -> DataFrame:
    """Volume filter, with time window."""
    samples = []
    if len(data_frame):
        timestamp_from = data_frame.iloc[0].timestamp
        # Iterator is not inclusive of timestamp_to, so increase by 1.
        timestamp_to = data_frame.iloc[-1].timestamp + pd.Timedelta("1t")
        if window:
            # Chunk data_frame by window.
            iterator = iter_window(timestamp_from, timestamp_to, window)
        else:
            iterator = iter_once(timestamp_from, timestamp_to)
        for ts_from, ts_to in iterator:
            df = filter_by_timestamp(data_frame, ts_from, ts_to)
            if len(df):
                next_index = 0
                df = df.reset_index()
                for row in df.itertuples():
                    index = row.Index
                    is_min_volume = row.volume >= min_volume if min_volume else True
                    if is_min_volume:
                        if index == 0:
                            sample = df.loc[:index]
                        else:
                            sample = df.loc[next_index:index]
                        samples.append(
                            volume_filter(sample, is_min_volume=is_min_volume)
                        )
                        next_index = index + 1
                total_rows = len(df)
                if next_index < total_rows:
                    sample = df.loc[next_index:]
                    samples.append(volume_filter(sample))
    filtered = pd.DataFrame(samples)
    # Assert data_frame volume is close to filtered volume.
    msg = "Volume is not close."
    assert is_decimal_close(data_frame.volume.sum(), filtered.totalVolume.sum()), msg
    return filtered


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
    buy_side = df[df.tickRule == 1]
    data.update(
        {
            "high": df.price.max(),
            "low": df.price.min(),
            "totalBuyVolume": buy_side.volume.sum() or ZERO,
            "totalVolume": df.volume.sum() or ZERO,
            "totalBuyNotional": buy_side.notional.sum() or ZERO,
            "totalNotional": df.notional.sum() or ZERO,
            "totalBuyTicks": buy_side.ticks.sum(),
            "totalTicks": df.ticks.sum(),
        }
    )
    return data


def cluster_trades_with_time_window(
    data_frame: DataFrame, window: str | None = None
) -> DataFrame:
    """Cluster trades, with time window."""
    data = []
    d = []
    direction = None
    for row in data_frame.itertuples():
        tick_rule = row.tickRule if row.tickRule in (1, -1) else None
        if direction is None:
            direction = tick_rule
            d.append(row)
        elif tick_rule == direction:
            d.append(row)
        else:
            direction = tick_rule
            data.append(d)
            d = [row]
    if d:
        data.append(d)
    return pd.DataFrame(cluster_trades(data))


def cluster_trades(data: list[dict]) -> list[dict]:
    """Cluster trades."""
    result = []
    for __, d in enumerate(data):
        first = d[0]
        last = d[-1]
        delta = last.timestamp - first.timestamp
        total_seconds = delta.total_seconds()
        values = [i.tickRule for i in d]
        is_up_tick = 1 in values
        is_down_tick = -1 in values
        assert not (is_up_tick and is_down_tick)
        if is_up_tick:
            tick_rule = 1
        elif is_down_tick:
            tick_rule = -1
        else:
            tick_rule = None
        data = {
            "timestamp": to_pydatetime(first.timestamp),
            "nanoseconds": first.nanoseconds,
            "total_seconds": total_seconds,
            "open": first.price,
            "high": max([i.price for i in d]),
            "low": min([i.price for i in d]),
            "close": last.price,
            "tickRule": tick_rule,
        }
        volume = ["volume", "totalBuyVolume", "totalVolume"]
        notional = ["notional", "totalBuyNotional", "totalNotional"]
        ticks = ["ticks", "totalBuyTicks", "totalTicks"]
        for sample_type in volume + notional + ticks:
            if all([hasattr(i, sample_type) for i in d]):
                value = sum(
                    [
                        val
                        for i in d
                        if (val := getattr(i, sample_type)) and not math.isnan(val)
                    ]
                )
                if sample_type in ticks:
                    value = int(value)
                data[sample_type] = value
        result.append(data)
    return result
