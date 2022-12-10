from quant_candles.utils import gettext_lazy as _

from ..trades import TradeData
from .base import Candle


def parse_thresh_attr(thresh_attr):
    thresh_attrs = ", ".join(THRESH_ATTRS)
    assert thresh_attr in THRESH_ATTRS, f"thresh_attr should be one of {thresh_attrs}"
    return thresh_attr


def parse_era_length(era_length):
    era_lengths = ", ".join(ERA_LENGTHS)
    assert era_length in ERA_LENGTHS, f"era_length should be one of {era_lengths}"
    return era_lengths


def get_initial_thresh_cache(thresh_attr, thresh_value, timestamp):
    return {
        "era": timestamp,
        "thresh_attr": thresh_attr,
        "thresh_value": thresh_value,
        "value": 0,
    }


def get_cache_for_era_length(cache, timestamp, era_length, thresh_attr, thresh_value):
    if not isinstance(cache["era"], datetime.date):
        date = cache["era"].date()
    else:
        date = cache["era"]
    next_date = timestamp.date()
    initial_cache = get_initial_thresh_cache(thresh_attr, thresh_value, timestamp)
    # Reset cache for new era
    if era_length == DAILY:
        if date != next_date:
            return initial_cache
    elif era_length == WEEKLY:
        if next_date.weekday() == 0:
            return initial_cache
    elif era_length == MONTHLY:
        if date.month != next_date.month:
            return initial_cache
    elif era_length == QUARTERLY:
        if pd.Timestamp(date).quarter != pd.Timestamp(next_date).quarter:
            return initial_cache
    else:
        raise NotImplementedError
    return cache


def merge_thresh_cache(previous, current, top_n=0):
    current["open"] = previous["open"]
    current["high"] = max(previous["high"], current["high"])
    current["low"] = min(previous["low"], current["low"])
    return merge_cache(previous, current, top_n=top_n)


def aggregate_thresh(data_frame, cache, thresh_attr, thresh_value, top_n=0):
    start = 0
    samples = []
    for index, row in data_frame.iterrows():
        cache[thresh_attr] += row[thresh_attr]
        if cache[thresh_attr] >= thresh_value:
            df = data_frame.loc[start:index]
            sample = aggregate_rows(df, uid=str(uuid4()), top_n=top_n)
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
    def on_trades(self, obj: TradeData) -> None:
        pass

    class Meta:
        proxy = True
        verbose_name = _("constant candle")
        verbose_name_plural = _("constant candles")
