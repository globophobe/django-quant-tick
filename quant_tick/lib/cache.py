from datetime import datetime

from pandas import DataFrame

from .candles import aggregate_candle


def get_next_cache(
    data_frame: DataFrame,
    cache_data: dict,
    timestamp: datetime | None = None,
) -> dict:
    """Get next cache."""
    values = aggregate_candle(data_frame, timestamp)
    if "next" in cache_data:
        previous_values = cache_data.pop("next")
        cache_data["next"] = merge_cache(previous_values, values)
    else:
        cache_data["next"] = values
    return cache_data


def merge_cache(previous: dict, current: dict) -> dict:
    """Merge cache."""
    current["timestamp"] = previous["timestamp"]
    current["open"] = previous["open"]
    if previous["high"] > current["high"]:
        current["high"] = previous["high"]
    if previous["low"] < current["low"]:
        current["low"] = previous["low"]
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
