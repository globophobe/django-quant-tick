from datetime import datetime
from decimal import Decimal
from typing import Optional

import pandas as pd
import pendulum
from pandas import DataFrame

from cryptofeed_werks.lib import aggregate_rows, iter_days, parse_period_from_to
from cryptofeed_werks.models import Candle, DailyTrades


def get_last_price(timestamp: datetime, default=None) -> Optional[Decimal]:
    """Get last price, or default."""
    open_price = Candle.objects.filter(timestamp__lt=timestamp).first()
    return open_price.values("close")["close"] if open_price else default


def aggregate(timestamp_from: datetime, timestamp_to: datetime, verbose: bool = False):
    candles = Candle.objects.filter(
        timestamp__gte=timestamp_from, timestamp_lt=timestamp_to, ok=True
    )
    timestamps = candles.values_list("timestamp", flat=True)
    missing = [
        timestamp
        for timestamp in pendulum.period(timestamp_from, timestamp_to).range("minutes")
        if timestamp not in timestamps
    ]
    dates = sorted(list(set([timestamp.date() for timestamp in missing])))
    for date in dates:
        timestamps_on_date = [
            timestamp for timestamp in missing if timestamp.date() == date
        ]
        # Iter minute
        if date == datetime.utcnow():
            ts_from = min(timestamps_on_date)
            ts_to = max(timestamps_on_date)
            data_frame = DailyTrades.get_data_frame(
                timestamp_from=ts_from, timestamp_to=ts_to
            )
        # Iter day
        else:
            for timestamp in iter_days(min(missing), max(missing)):
                date = timestamp.date()
                ts_from, ts_to = parse_period_from_to(date_from=date, date_to=date)
                data_frame = DailyTrades.get_data_frame(
                    timestamp_from=ts_from, timestamp_to=ts_to
                )
        update_candles(missing, data_frame)


def update_candles(self, timestamps: list, data_frame: DataFrame) -> None:
    candles = [{"timestamp": timestamp} for timestamp in timestamps]
    open_price = self.get_last_price(min(timestamps), default=data_frame.iloc[0].price)
    if data_frame is not None:
        for candle in candles:
            df = data_frame[
                (data_frame.timestamp >= candle["timestamp"])
                & (data_frame.timestamp < candle["timestamp"])
            ]
            if len(df):
                if "symbol" in df.columns:
                    symbols = df.symbol.unique()
                    assert len(symbols) == 1
                candle.update(
                    aggregate_rows(
                        df,
                        timestamp=candle["timestamp"],
                        open_price=open_price,
                    )
                )
                self.objects.update_or_create(candle)
                open_price = candle["open"]


def get_range(time_delta):
    total_seconds = time_delta.total_seconds()
    one_minute = 60.0
    one_hour = 3600.0
    if total_seconds < one_minute:
        unit = "seconds"
        step = total_seconds
    elif total_seconds >= one_minute and total_seconds <= one_hour:
        unit = "minutes"
        step = total_seconds / one_minute
    elif total_seconds > one_hour:
        unit = "hours"
        step = total_seconds / one_hour
    else:
        raise NotImplementedError
    step = int(step)
    assert total_seconds <= one_hour, f"{step} {unit} not supported"
    assert 60 % step == 0, f"{step} not divisible by 60"
    return unit, step


def aggregate_candles(
    data_frame,
    timestamp_from,
    timestamp_to,
    time_delta,
    top_n=0,
    cache={},
    as_dict=False,
    return_cache=True,
):
    samples = []
    period = pendulum.period(timestamp_from, timestamp_to)
    unit, step = get_range(time_delta)
    for index, start in enumerate(period.range(unit, step)):
        end = start + pd.Timedelta(f"{step} {unit}")
        df = data_frame[(data_frame.timestamp >= start) & (data_frame.timestamp < end)]
        # Maybe trades
        if len(df):
            if "symbol" in df.columns:
                symbols = df.symbol.unique()
                assert len(symbols) == 1
                open_price = cache[symbols[0]]
            else:
                open_price = cache["open"]
            sample = aggregate_rows(
                df,
                # Open timestamp, or won't be in partition
                timestamp=start,
                open_price=open_price,
                top_n=top_n,
            )
            cache["open"] = sample["close"]
            samples.append(sample)
    data = pd.DataFrame(samples) if not top_n and not as_dict else samples
    return data, cache if return_cache else data


class ClockTimeAggregator:
    def __init__(self, symbol, interval, **kwargs):
        self.symbol = symbol
        self.interval = interval
        self.timeframe = pd.Timedelta(interval)

    def process_data_frame(self, data_frame, cache):
        if self.futures:
            samples = []
            for symbol in data_frame.symbol.unique():
                df = data_frame[data_frame.symbol == symbol]
                data, cache = aggregate_candles(
                    df,
                    cache,
                    self.timestamp_from,
                    self.timestamp_to,
                    self.timeframe,
                    top_n=self.top_n,
                )
                samples.append(data)
            if all([isinstance(sample, pd.DataFrame) for sample in samples]):
                data = pd.concat(samples)
            else:
                data = samples
        else:
            data, cache = aggregate_candles(
                data_frame,
                cache,
                self.timestamp_from,
                self.timestamp_to,
                self.timeframe,
                top_n=self.top_n,
            )
        return data, cache
