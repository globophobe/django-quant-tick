import datetime
from decimal import Decimal
from operator import itemgetter
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pandas import DataFrame

from quant_candles.constants import SampleType

from .calendar import iter_once, iter_window, to_pydatetime
from .dataframe import is_decimal_close

ZERO = Decimal("0")


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
    assert is_decimal_close(
        data_frame.volume.sum(), aggregated.volume.sum()
    ), "Volume is not equal."
    return aggregated


def agg_trades(data_frame: DataFrame) -> Dict[str, Any]:
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
    data_frame: DataFrame, min_volume: int = 1000, window: Optional[str] = "1t"
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


def aggregate_sum(
    data_frame: DataFrame, attrs: Union[List[str], str] = None, window: str = "1t"
) -> DataFrame:
    """Aggregate sum over window."""
    samples = []
    if len(data_frame):
        if attrs is None:
            attrs = []
        elif not isinstance(attrs, list):
            attrs = [attrs]
        timestamp_from = data_frame.iloc[0].timestamp
        # Iterator is not inclusive of timestamp_to, so increase by 1.
        timestamp_to = data_frame.iloc[-1].timestamp + pd.Timedelta(window)
        for ts_from, ts_to in iter_window(timestamp_from, timestamp_to, value=window):
            df = data_frame[
                (data_frame.timestamp >= ts_from) & (data_frame.timestamp < ts_to)
            ]
            sample = {"timestamp": ts_from}
            for attr in attrs:
                sample[attr] = 0
            if len(df):
                for attr in attrs:
                    sample[attr] = df[attr].sum()
                samples.append(sample)
    return pd.DataFrame(samples).set_index("timestamp") if samples else pd.DataFrame()


def aggregate_candle(
    data_frame: DataFrame,
    timestamp: Optional[datetime.datetime] = None,
    sample_type: Optional[SampleType] = None,
    runs_n: Optional[int] = None,
    top_n: Optional[int] = None,
) -> dict:
    """Aggregate candle."""
    first_row = data_frame.iloc[0]
    last_row = data_frame.iloc[-1]
    high = data_frame.price.max()
    low = data_frame.price.min()
    data = {
        "timestamp": timestamp if timestamp else first_row.timestamp,
        "open": first_row.price,
        "high": high,
        "low": low,
        "close": last_row.price,
        "volume": data_frame.totalVolume.sum(),
        "buyVolume": data_frame.totalBuyVolume.sum(),
        "notional": data_frame.totalNotional.sum(),
        "buyNotional": data_frame.totalBuyNotional.sum(),
        "ticks": int(data_frame.totalTicks.sum()),
        "buyTicks": int(data_frame.totalBuyTicks.sum()),
    }
    if runs_n is not None:
        runs = get_runs(data_frame, runs_n)
        if runs:
            data["runs"] = runs
    if sample_type and top_n is not None:
        data["topN"] = get_top_n(data_frame, sample_type, top_n)
    return data


def get_top_n(data_frame: DataFrame, sample_type: SampleType, top_n: int) -> List[dict]:
    """Get top N."""
    # Filtered data may not have volume.
    index = (
        data_frame[data_frame.volume > 0][sample_type]
        .astype(float)
        .nlargest(top_n)
        .index
    )
    df = data_frame[data_frame.index.isin(index)]
    return get_records(df)


def get_records(df: DataFrame) -> List[dict]:
    """Get records."""
    data = df.to_dict("records")
    data.sort(key=itemgetter("timestamp", "nanoseconds"))
    for item in data:
        for key in list(item):
            if key not in (
                "exchange",
                "symbol",
                "timestamp",
                "nanoseconds",
                "price",
                "volume",
                "notional",
                "ticks",
                "tickRule",
            ):
                del item[key]
    return data


def get_runs(
    data_frame: DataFrame,
    runs_n: int = 0,
    bins: List[int] = [1_000, 10_000, 50_000, 100_000],
) -> List[dict]:
    """Get runs."""
    runs = []
    run = []
    direction = None
    # Filtered data may not have volume.
    df = data_frame[data_frame.volume > 0]
    for row in df.itertuples():
        if direction is None:
            direction = row.tickRule
            run.append(row)
        elif row.tickRule == direction:
            run.append(row)
        else:
            direction = row.tickRule
            if len(run) >= runs_n:
                runs.append(run)
            run = [row]
    if run and len(run) >= runs_n:
        runs.append(run)
    # Result.
    result = []
    for index, run in enumerate(runs):
        timestamp = run[0].timestamp
        delta = run[-1].timestamp - timestamp
        data = {
            "timestamp": to_pydatetime(timestamp),
            "duration": delta.total_seconds() or 0,
        }
        for sample_type in SampleType.values:
            value = sum([getattr(r, sample_type) for r in run])
            if sample_type == SampleType.TICK.value:
                value = int(value)
            data[sample_type] = value * int(run[0].tickRule)
        data.update(get_bins(run, bins))
        result.append(data)
    return result


def get_bins(run: List[tuple], bins: List[int]):
    """Get bins."""
    data = {}
    for index, b in enumerate(bins):
        next_index = index + 1
        c = f"volume{b}"
        if (next_index) < len(bins):
            next_b = bins[next_index]
            data[c] = len([r for r in run if r.volume >= b and r.volume < next_b])
        else:
            data[c] = len([r for r in run if r.volume >= b])
    return data
