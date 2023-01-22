from operator import itemgetter
from typing import Optional

from pandas import DataFrame

from quant_candles.constants import SampleType

from .aggregate import aggregate_candle


def get_next_cache(
    data_frame: DataFrame,
    cache_data: dict,
    sample_type: Optional[SampleType] = None,
    runs_n: Optional[int] = None,
    top_n: Optional[int] = None,
) -> dict:
    """Get next cache."""
    values = aggregate_candle(
        data_frame, sample_type=sample_type, runs_n=runs_n, top_n=top_n
    )
    if "next" in cache_data:
        previous_values = cache_data.pop("next")
        cache_data["next"] = merge_cache(
            previous_values, values, sample_type, runs_n, top_n
        )
    else:
        cache_data["next"] = values
    return cache_data


def merge_cache(
    previous: dict,
    current: dict,
    sample_type: SampleType,
    runs_n: Optional[int] = None,
    top_n: Optional[int] = None,
) -> dict:
    """Merge cache."""
    for key in (
        "volume",
        "buyVolume",
        "notional",
        "buyNotional",
        "ticks",
        "buyTicks",
    ):
        current[key] += previous[key]  # Add
    if runs_n is not None:
        current["runs"] = previous.get("runs", []) + current.get("runs", [])
    if top_n is not None:
        merged_top = previous.get("topN", []) + current.get("topN", [])
        if len(merged_top):
            # Sort by sample_type
            merged_top.sort(key=lambda x: x[sample_type], reverse=True)
            # Slice top_n
            m = merged_top[:top_n]
            # Sort by timestamp, nanoseconds
            m.sort(key=itemgetter("timestamp", "nanoseconds"))
            current["topN"] = m
    return current
