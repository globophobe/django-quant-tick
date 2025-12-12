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
    current["timestamp"] = previous["timestamp"]

    # Single-exchange: merge top-level OHLC
    if "open" in previous and "open" in current:
        current_open = current.get("open")
        current["open"] = previous["open"]
        if previous["high"] > current["high"]:
            current["high"] = previous["high"]
        if previous["low"] < current["low"]:
            current["low"] = previous["low"]
        # Realized variance with cross-segment
        if "realizedVariance" in current and "realizedVariance" in previous:
            cross_var = _calc_cross_segment_variance(
                previous.get("close"), current_open
            )
            current["realizedVariance"] += previous["realizedVariance"] + cross_var

    # Sum aggregate volume fields
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
    # Exchange
    for exchange in _get_exchanges(previous):
        if exchange in _get_exchanges(current):
            _merge_exchange(previous, current, exchange)
    return current


def _get_exchanges(data: dict) -> set[str]:
    """Extract exchange names."""
    exchanges = set()
    for key in data:
        if key.endswith("Open"):
            exchanges.add(key[:-4])
    return exchanges


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


def _merge_exchange(previous: dict, current: dict, exchange: str) -> None:
    """Merge per-exchange fields in place."""
    # OHLC
    open_key = f"{exchange}Open"
    high_key = f"{exchange}High"
    low_key = f"{exchange}Low"
    close_key = f"{exchange}Close"

    curr_open = current.get(open_key)
    current[open_key] = previous[open_key]
    if previous[high_key] > current[high_key]:
        current[high_key] = previous[high_key]
    if previous[low_key] < current[low_key]:
        current[low_key] = previous[low_key]
    # Volume
    volume_key = f"{exchange}Volume"
    current[volume_key] += previous[volume_key]
    # Realized variance with cross-segment
    var_key = f"{exchange}RealizedVariance"
    prev_close = previous.get(close_key)
    cross_var = _calc_cross_segment_variance(prev_close, curr_open)
    current[var_key] += previous[var_key] + cross_var
