from datetime import UTC, datetime
from functools import partial

import pandas as pd
from pandas import DataFrame

from quant_tick.controllers import iter_api
from quant_tick.lib import (
    candles_to_data_frame,
    parse_fixed_resolution_minutes,
    resample_candles,
)

from .api import (
    format_bitmex_api_timestamp,
    get_bitmex_api_pagination_id,
    get_bitmex_api_timestamp,
    get_bitmex_api_url,
)
from .constants import API_URL, MAX_RESULTS, MIN_ELAPSED_PER_REQUEST
from .trades import get_bitmex_api_response

BITMEX_BIN_SIZE_BY_MINUTES = {
    1: "1m",
    5: "5m",
    60: "1h",
    1440: "1d",
}


def parse_bitmex_candle_resolution(value: str | int | None) -> int:
    """Parse BitMEX candle resolution into whole minutes."""
    return parse_fixed_resolution_minutes(value)


def get_bitmex_fetch_bin_size(resolution: str | int | None) -> tuple[int, str]:
    """Get the largest BitMEX bin size that can be resampled."""
    target_minutes = parse_bitmex_candle_resolution(resolution)
    if target_minutes in BITMEX_BIN_SIZE_BY_MINUTES:
        return target_minutes, BITMEX_BIN_SIZE_BY_MINUTES[target_minutes]
    if target_minutes % 60 == 0:
        return target_minutes, BITMEX_BIN_SIZE_BY_MINUTES[60]
    if target_minutes % 5 == 0:
        return target_minutes, BITMEX_BIN_SIZE_BY_MINUTES[5]
    return target_minutes, BITMEX_BIN_SIZE_BY_MINUTES[1]


def resample_bitmex_candles(
    data_frame: DataFrame,
    *,
    timestamp_from: datetime,
    timestamp_to: datetime,
    resolution_minutes: int,
) -> DataFrame:
    """Aggregate BitMEX candles to the requested resolution."""
    return resample_candles(
        data_frame,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        resolution_minutes=resolution_minutes,
    )


def fetch_bitmex_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    *,
    bin_size: str,
    log_format: str | None = None,
) -> DataFrame:
    """Fetch BitMEX candles."""
    fetch_minutes = parse_bitmex_candle_resolution(bin_size)
    # BitMEX bucket timestamps are candle close timestamps.
    ts_from = timestamp_from + pd.Timedelta(f"{fetch_minutes}min")
    start_time = format_bitmex_api_timestamp(ts_from)
    params = f"symbol={api_symbol}&startTime={start_time}&binSize={bin_size}"
    url = f"{API_URL}/trade/bucketed?{params}"
    candles, _, _ = iter_api(
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
    candles = dedupe_candles(candles)
    return candles_to_data_frame(timestamp_from, timestamp_to, candles)


def bitmex_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    bin_size: str = "1m",
    resolution: str | int | None = None,
    log_format: str | None = None,
) -> DataFrame:
    """Get candles."""
    requested = resolution if resolution is not None else bin_size
    target_minutes, fetch_bin_size = get_bitmex_fetch_bin_size(requested)
    data_frame = fetch_bitmex_candles(
        api_symbol,
        timestamp_from,
        timestamp_to,
        bin_size=fetch_bin_size,
        log_format=log_format,
    )
    if parse_bitmex_candle_resolution(fetch_bin_size) == target_minutes:
        return data_frame
    return resample_bitmex_candles(
        data_frame,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        resolution_minutes=target_minutes,
    )


def dedupe_candles(candles: list[dict]) -> list[dict]:
    """Drop duplicates.

    BitMEX sometimes returns multiple candles for the same timestamp; keep one only
    if the duplicate payloads are identical.
    """
    deduped = []
    by_timestamp = {}
    for candle in candles:
        timestamp = candle["timestamp"]
        existing = by_timestamp.get(timestamp)
        if existing is None:
            by_timestamp[timestamp] = candle
            deduped.append(candle)
        elif candle != existing:
            raise ValueError("BitMEX returned conflicting duplicate candles.")
    return deduped
