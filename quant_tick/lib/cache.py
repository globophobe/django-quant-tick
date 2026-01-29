import math
from datetime import datetime
from decimal import Decimal

from pandas import DataFrame

from .candles import aggregate_candle

ZERO = Decimal("0")


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
    curr_open = current.get("open")
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

    cross_var = _calc_cross_segment_variance(previous.get("close"), curr_open)
    current["realizedVariance"] += previous["realizedVariance"] + cross_var

    prev_dist = previous.get("distribution")
    curr_dist = current.get("distribution")
    if prev_dist or curr_dist:
        current["distribution"] = _merge_distributions(prev_dist or {}, curr_dist or {})

    return current


def _calc_cross_segment_variance(
    prev_close: Decimal | None, curr_open: Decimal | None
) -> Decimal:
    """Calculate cross-segment variance from log return squared."""
    if prev_close is None or curr_open is None:
        return Decimal("0")
    prev_close_float = float(prev_close)
    curr_open_float = float(curr_open)
    if prev_close_float > 0 and curr_open_float > 0:
        log_ret = math.log(curr_open_float) - math.log(prev_close_float)
        return Decimal(str(log_ret**2))
    return Decimal("0")


def _merge_distributions(prev: dict, curr: dict) -> dict:
    """Merge two distribution dicts by summing metrics per level."""
    merged = {}
    for level in set(prev.keys()) | set(curr.keys()):
        p = prev.get(level, {})
        c = curr.get(level, {})
        merged[level] = {
            "ticks": p.get("ticks", 0) + c.get("ticks", 0),
            "buyTicks": p.get("buyTicks", 0) + c.get("buyTicks", 0),
            "volume": p.get("volume", ZERO) + c.get("volume", ZERO),
            "buyVolume": p.get("buyVolume", ZERO) + c.get("buyVolume", ZERO),
            "notional": p.get("notional", ZERO) + c.get("notional", ZERO),
            "buyNotional": p.get("buyNotional", ZERO) + c.get("buyNotional", ZERO),
        }
    return merged
