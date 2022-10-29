from datetime import datetime, timezone
from functools import partial
from typing import List, Optional

import pandas as pd

from quant_candles.controllers import iter_api
from quant_candles.lib import candles_to_data_frame

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
    log_format: Optional[str] = None,
) -> List[dict]:
    """Get candles."""
    # Timestamp is at candle close.
    ts_from = timestamp_from + pd.Timedelta(value="1t")
    start_time = format_bitmex_api_timestamp(ts_from)
    params = f"symbol={api_symbol}&startTime={start_time}&binSize={bin_size}"
    url = f"{API_URL}/trade/bucketed/?{params}"
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
        tstamp = datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc)
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
            del candle[key]
    return candles_to_data_frame(timestamp_from, timestamp_to, candles)
