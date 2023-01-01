import datetime
from typing import Iterable

from pandas import DataFrame

from quant_candles.constants import Frequency
from quant_candles.lib import aggregate_rows, get_next_cache, merge_cache
from quant_candles.utils import gettext_lazy as _

from ..candles import Candle
from ..trades import TradeData


def get_range(time_delta):
    """Get range."""
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


def get_initial_thresh_cache(thresh_attr, thresh_value, timestamp):
    return {
        "era": timestamp,
        "thresh_attr": thresh_attr,
        "thresh_value": thresh_value,
        "value": 0,
    }


def get_cache_for_frequency(
    cache: dict, timestamp: datetime.datetime, era_length, thresh_attr, thresh_value
):
    if not isinstance(cache["era"], datetime.date):
        date = cache["era"].date()
    else:
        date = cache["era"]
    next_date = timestamp.date()
    initial_cache = get_initial_thresh_cache(thresh_attr, thresh_value, timestamp)
    # Reset cache for new era
    if era_length == Frequency.DAILY:
        if date != next_date:
            return initial_cache
    elif era_length == Frequency.WEEKLY:
        if next_date.weekday() == 0:
            return initial_cache
    else:
        raise NotImplementedError
    return cache


def merge_thresh_cache(previous, current, top_n=0):
    current["open"] = previous["open"]
    current["high"] = max(previous["high"], current["high"])
    current["low"] = min(previous["low"], current["low"])
    return merge_cache(previous, current, top_n=top_n)


def aggregate_thresh(
    data_frame: DataFrame,
    cache: dict,
    thresh_attr: str,
    thresh_value: int,
    top_n: int = 0,
):
    """Aggregate thresh."""
    start = 0
    samples = []
    for index, row in data_frame.iterrows():
        cache[thresh_attr] += row[thresh_attr]
        if cache[thresh_attr] >= thresh_value:
            df = data_frame.loc[start:index]
            sample = aggregate_rows(df)
            if "nextDay" in cache:
                previous = cache.pop("nextDay")
                sample = merge_cache(previous, sample, top_n=top_n)
            samples.append(sample)
            # Reinitialize cache
            cache[thresh_attr] = 0
            # Next index
            start = index + 1
    # Cache
    is_last_row = start == len(data_frame)
    if not is_last_row:
        df = data_frame.loc[start:]
        cache = get_next_cache(df, cache, top_n=top_n)
    return samples, cache


class ConstantCandle(Candle):
    @classmethod
    def on_trades(cls, objs: Iterable[TradeData]) -> None:
        """On trades."""
        pass

    class Meta:
        proxy = True
        verbose_name = _("constant candle")
        verbose_name_plural = _("constant candles")
