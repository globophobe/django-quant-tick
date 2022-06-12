from datetime import datetime, timezone
from functools import partial
from typing import List, Optional

from cryptofeed_werks.controllers import iter_api
from cryptofeed_werks.lib import (
    candles_to_data_frame,
    parse_datetime,
    timestamp_to_inclusive,
)

from .api import format_ftx_api_timestamp, get_ftx_api_response
from .constants import API_URL, MAX_RESULTS, MIN_ELAPSED_PER_REQUEST


def get_ftx_candle_url(
    url: str, timestamp_from: datetime, pagination_id: Optional[str] = None
) -> str:
    """Get FTX candle URL."""
    start_time = format_ftx_api_timestamp(timestamp_from)
    url += f"&start_time={start_time}"
    if pagination_id:
        return url + f"&end_time={pagination_id}"
    return url


def get_ftx_candle_pagination_id(
    timestamp: datetime, last_data: List[dict] = [], data: List[dict] = []
) -> Optional[float]:
    """Get FTX candle pagination_id.

    Pagination details: https://docs.ftx.com/#get-historical-prices
    """
    if len(data):
        return data[-1]["time"]


def get_ftx_candle_timestamp(candle) -> datetime:
    """Get FTX candle timestamp."""
    return parse_datetime(candle["startTime"])


def ftx_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    resolution: int = 60,
    log_format: Optional[str] = None,
):
    """Get candles."""
    ts_to = timestamp_to_inclusive(timestamp_from, timestamp_to, value="1t")
    url = f"{API_URL}/markets/{api_symbol}/candles?resolution={resolution}"
    candles, _ = iter_api(
        url,
        get_ftx_candle_pagination_id,
        get_ftx_candle_timestamp,
        partial(get_ftx_api_response, get_ftx_candle_url),
        MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=format_ftx_api_timestamp(ts_to),
        log_format=log_format,
    )
    for candle in candles:
        candle["timestamp"] = datetime.fromisoformat(candle["startTime"]).replace(
            tzinfo=timezone.utc
        )
        for key in ("startTime", "time"):
            del candle[key]
    return candles_to_data_frame(timestamp_from, timestamp_to, candles)
