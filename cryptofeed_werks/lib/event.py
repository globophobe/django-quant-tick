from uuid import uuid4

from cryptofeed_werks.constants import SeriesType
from cryptofeed_werks.lib import aggregate_rows


def get_initial_thresh_cache(thresh_attr, thresh_value, timestamp):
    return {
        "thresh_attr": thresh_attr,
        "thresh_value": thresh_value,
        "value": 0,
    }


def get_next_cache(cache: dict, next_day: {}) -> dict:
    """Get next cache."""
    if "nextDay" in cache:
        previous_day = cache.pop("nextDay")
        cache["nextDay"] = merge_cache(previous_day, next_day)
    else:
        cache["nextDay"] = next_day
    return cache


def merge_cache(previous: dict, current: dict) -> dict:
    """Merge cache."""
    current["open"] = previous["open"]
    current["high"] = max(previous["high"], current["high"])
    current["low"] = min(previous["low"], current["low"])
    for key in (
        "volume",
        "buyVolume",
        "notional",
        "buyNotional",
        "ticks",
        "buyTicks",
    ):
        current[key] += previous[key]  # Add
    return current


def aggregate_thresh(data_frame, cache, thresh_attr, thresh_value):
    start = 0
    samples = []
    for index, row in data_frame.iterrows():
        cache[thresh_attr] += row[thresh_attr]
        if cache[thresh_attr] >= thresh_value:
            df = data_frame.loc[start:index]
            sample = aggregate_rows(df, uid=str(uuid4()))
            if "nextDay" in cache:
                previous = cache.pop("nextDay")
                sample = merge_cache(previous, sample)
            samples.append(sample)
            # Reinitialize cache
            cache[thresh_attr] = 0
            # Next index
            start = index + 1
    # Cache
    is_last_row = start == len(data_frame)
    if not is_last_row:
        df = data_frame.loc[start:]
        cache = get_next_cache(df, cache)
    return samples, cache


class EventTimeAggregator:
    def __init__(
        self,
        source_table: str,
        series_type: SeriesType,
        interval: str = None,
        top_n: int = 0,
        **kwargs,
    ):
        if series_type == SeriesType.TICK:
            self.thresh_attr = SeriesType.TICK + "s"
        else:
            self.thresh_attr = series_type
        self.thresh_value = int(interval)
        destination_table = (
            f"{source_table}_{self.thresh_attr}{self.thresh_value}{self.cache_reset}"
        )
        if top_n:
            destination_table += f"_top{top_n}"
        super().__init__(source_table, destination_table, **kwargs)
        self.top_n = top_n

    def process_data_frame(self, data_frame, cache):
        return aggregate_thresh(
            data_frame,
            cache,
            self.thresh_attr,
            self.thresh_value,
            top_n=self.top_n,
        )
