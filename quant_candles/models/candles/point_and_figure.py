from decimal import Decimal

import numpy as np
import pandas as pd

from ...bqloader import MULTIPLE_SYMBOL_RENKO_SCHEMA, SINGLE_SYMBOL_RENKO_SCHEMA
from ...controllers import get_next_cache, get_top_n, merge_cache, parse_period_from_to
from ...fscache import firestore_data
from ..base import BaseCacheAggregator
from ..lib import get_decimal_value_for_table_id, get_source_table
from .lib import aggregate_renko, get_initial_cache


def get_level(price, box_size):
    return int(price / box_size) * box_size


def get_initial_cache(data_frame, box_size):
    row = data_frame.loc[0]
    # First trade decides level, so is discarded
    df = data_frame.loc[1:]
    df.reset_index(drop=True, inplace=True)
    cache = {"level": get_level(row.price, box_size), "direction": None}
    return df, cache


def update_cache(cache, level, change):
    cache["level"] = level
    cache["direction"] = np.sign(change)
    return cache


def get_bounds(cache, box_size, reversal=1):
    level = cache["level"]
    direction = cache["direction"]
    if direction == 1:
        high = level + box_size
        low = level - (box_size * reversal)
    elif direction == -1:
        high = level + (box_size * reversal)
        low = level - box_size
    else:
        high = level + box_size
        low = level - box_size
    return high, low


def get_change(cache, high, low, price, box_size, level_func=get_level):
    level = cache["level"]
    higher = price >= high
    lower = price < low
    if higher or lower:
        current_level = level_func(price, box_size)
        change = current_level - level
        # Did price break below threshold?
        if lower:
            # Is there a remainder?
            if price % box_size != 0:
                change += box_size
                current_level += box_size
        return current_level, change
    return level, 0


def aggregate_renko(data_frame, cache, box_size, top_n=10, level_func=get_level):
    start = 0
    samples = []
    high, low = get_bounds(cache, box_size)
    for index, row in data_frame.iterrows():
        level, change = get_change(
            cache, high, low, row.price, box_size, level_func=level_func
        )
        if change:
            df = data_frame.loc[start:index]
            sample = aggregate(df, level, top_n=top_n)
            if "nextDay" in cache:
                # Next day is today's previous
                previous_day = cache.pop("nextDay")
                sample = merge_cache(previous_day, sample, top_n=top_n)
            # Is new level higher or lower than previous?
            assert_higher_or_lower(level, cache)
            # Is price bounded by the next higher or lower level?
            assert_bounds(row, cache, box_size)
            # Next index
            start = index + 1
            # Update cache
            cache = update_cache(cache, level, change)
            high, low = get_bounds(cache, box_size)
            samples.append(sample)
    # Cache
    is_last_row = start == len(data_frame)
    if not is_last_row:
        next_day = aggregate(data_frame.loc[start:], level, top_n=top_n)
        cache = get_next_cache(cache, next_day, top_n=top_n)
    return samples, cache


def aggregate(df, level, top_n=0):
    first_row = df.iloc[0]
    last_row = df.iloc[-1]
    buy_side = df[df.tickRule == 1]
    data = {
        # Close timestamp, or won't be in partition
        "timestamp": last_row.timestamp,
        "nanoseconds": last_row.nanoseconds,
        "level": level,
        "price": last_row.price,
        "buyVolume": buy_side.volume.sum(),
        "volume": df.volume.sum(),
        "buyNotional": buy_side.notional.sum(),
        "notional": df.notional.sum(),
        "buyTicks": buy_side.ticks.sum(),
        "ticks": df.ticks.sum(),
    }
    if "symbol" in df.columns:
        assert len(df.symbol.unique()) == 1
        data["symbol"] = first_row.symbol
    if top_n:
        data["topN"] = get_top_n(df, top_n=top_n)
    return data


def assert_higher_or_lower(level, cache):
    assert level < cache["level"] or level > cache["level"]


def assert_bounds(row, cache, box_size):
    high, low = get_bounds(cache, box_size)
    assert low <= cache["level"] <= high
    assert high == low + (box_size * 2)


def renko_aggregator(
    provider: str,
    symbol: str,
    period_from: str,
    period_to: str,
    box_size: str,
    top_n: int = 0,
    futures: bool = False,
    verbose: bool = False,
):
    assert box_size, 'Required param "box_size" not provided'
    date_from, date_to = parse_period_from_to(
        period_from=period_from, period_to=period_to
    )
    # Reversed, daily then hourly
    RenkoAggregator(
        get_source_table(provider, symbol, futures=futures),
        period_from=date_from,
        period_to=date_to,
        futures=futures,
        box_size=box_size,
        top_n=top_n,
        verbose=verbose,
    ).main()


class RenkoAggregator(BaseCacheAggregator):
    def __init__(self, source_table, box_size, reversal=1, top_n=0, **kwargs):
        self.box_size = Decimal(box_size)
        box = get_decimal_value_for_table_id(self.box_size)
        destination_table = f"{source_table}_renko{box}"
        if top_n:
            destination_table += f"_top{top_n}"
        super().__init__(source_table, destination_table, **kwargs)
        self.reversal = reversal
        self.top_n = top_n

    @property
    def schema(self):
        if self.futures:
            return MULTIPLE_SYMBOL_RENKO_SCHEMA
        else:
            return SINGLE_SYMBOL_RENKO_SCHEMA

    def get_initial_cache(self, data_frame):
        return get_initial_cache(data_frame, self.box_size)

    def get_cache(self, data_frame):
        data_frame, data = super().get_cache(data_frame)
        data = firestore_data(data, deserialize=True)
        return data_frame, data

    def process_data_frame(self, data_frame, cache):
        if self.futures:
            samples = []
            for symbol in data_frame.symbol.unique():
                df = data_frame[data_frame.symbol == symbol]
                data, cache = aggregate_renko(
                    df,
                    cache,
                    self.box_size,
                    top_n=self.top_n,
                )
                samples.append(data)
            if all([isinstance(sample, pd.DataFrame) for sample in samples]):
                data = pd.concat(samples)
            else:
                data = samples
        else:
            data, cache = aggregate_renko(
                data_frame,
                cache,
                self.box_size,
                top_n=self.top_n,
            )
        return data, cache
