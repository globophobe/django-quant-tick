from datetime import datetime, timezone
from decimal import Decimal
from functools import partial

from cryptofeed_werks.controllers import iter_api

from .api import get_coinbase_api_response
from .constants import API_URL, MAX_RESULTS, MIN_ELAPSED_PER_REQUEST


def get_coinbase_candle_url(url, timestamp_from, pagination_id):
    """Get Coinbase candle URL."""
    start = timestamp_from.replace(tzinfo=None).isoformat()
    url += f"&start={start}"
    if pagination_id:
        url += f"&end={pagination_id}"
    return url


def get_coinbase_candle_pagination_id(timestamp, last_data=[], data=[]):
    """Get Coinbase candle pagination_id.

    Pagination details: https://docs.pro.coinbase.com/#pagination
    """
    if len(data):
        return datetime.fromtimestamp(data[-1][0]).isoformat()


def get_coinbase_candle_timestamp(candle):
    """Get Coinbase candle timestamp."""
    return datetime.fromtimestamp(candle[0]).replace(tzinfo=timezone.utc)


def get_candles(symbol, timestamp_from, timestamp_to, granularity=60, log_format=None):
    url = f"{API_URL}/products/{symbol}/candles?granularity={granularity}"
    candles, _ = iter_api(
        url,
        get_coinbase_candle_pagination_id,
        get_coinbase_candle_timestamp,
        partial(get_coinbase_api_response, get_coinbase_candle_url),
        MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=timestamp_to,
        log_format=log_format,
    )
    return [
        {
            "timestamp": datetime.fromtimestamp(candle[0]).replace(tzinfo=timezone.utc),
            "open": Decimal(str(candle[3])),
            "high": Decimal(str(candle[2])),
            "low": Decimal(str(candle[1])),
            "close": Decimal(str(candle[4])),
            "notional": Decimal(str(candle[5])),
        }
        for candle in candles
    ]
