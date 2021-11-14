import json
import time
from decimal import Decimal

import httpx

from cryptofeed_werks.controllers import (
    HTTPX_ERRORS,
    increment_api_total_requests,
    iter_api,
    throttle_api_requests,
)
from cryptofeed_werks.lib import parse_datetime

from .constants import (
    API_URL,
    BITFINEX_MAX_REQUESTS_RESET,
    BITFINEX_TOTAL_REQUESTS,
    MAX_REQUESTS,
    MAX_REQUESTS_RESET,
    MAX_RESULTS,
    MIN_ELAPSED_PER_REQUEST,
)


def get_bitfinex_api_url(url, pagination_id):
    if pagination_id:
        return url + f"&end={pagination_id}"
    return url


def get_bitfinex_api_pagination_id(timestamp, last_data=[], data=[]):
    if len(data):
        return data[-1][1]


def get_bitfinex_api_timestamp(trade):
    return parse_datetime(trade[1], unit="ms")


def get_trades(symbol, timestamp_from, pagination_id, log_format=None):
    # No start query param
    # Specifying start, end returns MAX_RESULTS
    url = f"{API_URL}/{symbol}/hist?limit={MAX_RESULTS}"
    return iter_api(
        url,
        get_bitfinex_api_pagination_id,
        get_bitfinex_api_timestamp,
        get_bitfinex_api_response,
        MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=pagination_id,
        log_format=log_format,
    )


def get_bitfinex_api_response(url, pagination_id=None, retry=30):
    throttle_api_requests(
        BITFINEX_MAX_REQUESTS_RESET,
        BITFINEX_TOTAL_REQUESTS,
        MAX_REQUESTS_RESET,
        MAX_REQUESTS,
    )
    try:
        response = httpx.get(get_bitfinex_api_url(url, pagination_id))
        increment_api_total_requests(BITFINEX_TOTAL_REQUESTS)
        if response.status_code == 200:
            result = response.read()
            return json.loads(result, parse_float=Decimal)
        else:
            response.raise_for_status()
    except HTTPX_ERRORS:
        if retry > 0:
            time.sleep(1)
            retry -= 1
            return get_bitfinex_api_response(url, pagination_id, retry)
        raise
