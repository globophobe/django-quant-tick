from datetime import UTC, datetime
from functools import partial

import pandas as pd
from pandas import DataFrame

from quant_tick.controllers import iter_api
from quant_tick.lib import candles_to_data_frame

from .api import (
    format_bitmex_api_timestamp,
    get_bitmex_api_pagination_id,
    get_bitmex_api_timestamp,
    get_bitmex_api_url,
)
from .constants import API_URL, MAX_RESULTS, MIN_ELAPSED_PER_REQUEST
from .trades import get_bitmex_api_response


def bitmex_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    bin_size: str = "1m",
    log_format: str | None = None,
) -> DataFrame:
    """Get candles."""
    # Timestamp is at candle close.
    ts_from = timestamp_from + pd.Timedelta(value="1min")
    start_time = format_bitmex_api_timestamp(ts_from)
    params = f"symbol={api_symbol}&startTime={start_time}&binSize={bin_size}"
    url = f"{API_URL}/trade/bucketed?{params}"
    candles, _ = iter_api(
        url,
        get_bitmex_api_pagination_id,
        get_bitmex_api_timestamp,
        partial(get_bitmex_api_response, get_bitmex_api_url),
        MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=format_bitmex_api_timestamp(timestamp_to),
        log_format=log_format,
    )
    for candle in candles:
        timestamp, _ = candle["timestamp"].split(".000Z")
        tstamp = datetime.fromisoformat(timestamp).replace(tzinfo=UTC)
        candle["timestamp"] = tstamp - pd.Timedelta(bin_size)
        candle["volume"] = candle["foreignNotional"]
        for key in (
            "symbol",
            "vwap",
            "lastSize",
            "turnover",
            "homeNotional",
            "foreignNotional",
        ):
            if key in candle:
                del candle[key]
    # In rare cases, Bitmex API may return duplicates.
    delta = timestamp_to - timestamp_from
    expected = int(delta.total_seconds() / 60)
    if len(candles) > expected:
        timestamps = set()
        duplicates = {}
        for candle in candles:
            timestamp = candle["timestamp"]
            if timestamp in timestamps:
                duplicates.setdefault(timestamp, []).append(candle)
            else:
                timestamps.add(timestamp)
        for timestamp in duplicates:
            dups = duplicates[timestamp]
            first = dups[0]
            if all([d for d in dups if d == first]):
                match = False
                should_remove = []
                for index, candle in enumerate(candles):
                    is_equal = candle == first
                    if not match and is_equal:
                        match = True
                    elif is_equal:
                        should_remove.append(index)
                candles = [
                    candle
                    for index, candle in enumerate(candles)
                    if index not in should_remove
                ]
    return candles_to_data_frame(timestamp_from, timestamp_to, candles)
