from quant_candles.controllers import iter_api
from quant_candles.lib import parse_datetime

from .api import get_bitmex_api_pagination_id, get_bitmex_api_response
from .constants import API_URL, MAX_RESULTS, MIN_ELAPSED_PER_REQUEST


def get_funding(symbol, timestamp_from, pagination_id, log_format=None):
    url = f"{API_URL}/funding?symbol={symbol}"
    timestamp_key = "timestamp"
    return [
        {
            "timestamp": parse_datetime(f["timestamp"]),
            "rate": f["fundingRate"],
        }
        for f in iter_api(
            url,
            timestamp_key,
            get_bitmex_api_pagination_id,
            get_bitmex_api_response,
            parse_datetime,
            MAX_RESULTS,
            MIN_ELAPSED_PER_REQUEST,
            timestamp_from=timestamp_from,
            pagination_id=pagination_id,
            log_format=log_format,
        )
    ]
