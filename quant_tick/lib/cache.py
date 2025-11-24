import math
from datetime import datetime
from decimal import Decimal

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
    # Get current open for cross-segment variance)
    current_open = current.get("open")
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
        "roundVolume",
        "roundBuyVolume",
        "roundNotional",
        "roundBuyNotional",
    ):
        current[key] += previous[key]
    # Realized variance: add cross-segment log return squared
    prev_close = previous.get("close")
    cross_segment_var = Decimal("0")
    if prev_close is not None and current_open is not None:
        prev_close_float = float(prev_close)
        curr_open_float = float(current_open)
        if prev_close_float > 0 and curr_open_float > 0:
            log_ret = math.log(curr_open_float) - math.log(prev_close_float)
            cross_segment_var = Decimal(str(log_ret**2))
    current["realizedVariance"] += previous["realizedVariance"] + cross_segment_var
    return current
