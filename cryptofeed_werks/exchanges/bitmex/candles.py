from datetime import datetime, timezone
from functools import partial
from typing import List, Optional

import pandas as pd

from cryptofeed_werks.controllers import iter_api

from .api import (
    format_bitmex_api_timestamp,
    get_bitmex_api_pagination_id,
    get_bitmex_api_timestamp,
    get_bitmex_api_url,
)
from .constants import API_URL, MAX_RESULTS, MIN_ELAPSED_PER_REQUEST
from .trades import get_bitmex_api_response


def get_candles(
    symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    bin_size: str = "1m",
    log_format: Optional[str] = None,
) -> List[dict]:
    """Get candles."""
    start_time = format_bitmex_api_timestamp(timestamp_from)
    params = f"symbol={symbol}&startTime={start_time}&binSize={bin_size}"
    url = f"{API_URL}/trade/bucketed/?{params}"
    candles, _ = iter_api(
        url,
        get_bitmex_api_pagination_id,
        get_bitmex_api_timestamp,
        partial(get_bitmex_api_response, get_bitmex_api_url),
        MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=timestamp_to,
        log_format=log_format,
    )
    for candle in candles:
        timestamp, _ = candle["timestamp"].split(".000Z")
        tstamp = datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc)
        candle["timestamp"] = tstamp - pd.Timedelta(bin_size)
        for key in (
            "symbol",
            "vwap",
            "lastSize",
            "turnover",
            "homeNotional",
            "foreignNotional",
        ):
            del candle[key]
    return candles
