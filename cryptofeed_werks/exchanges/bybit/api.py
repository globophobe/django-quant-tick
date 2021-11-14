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
    BYBIT_MAX_REQUESTS_RESET,
    BYBIT_TOTAL_REQUESTS,
    MAX_REQUESTS,
    MAX_REQUESTS_RESET,
    MAX_RESULTS,
    MIN_ELAPSED_PER_REQUEST,
)


def get_bybit_api_url(url, pagination_id):
    if pagination_id:
        return url + f"&from={pagination_id}"
    return url


def get_bybit_api_pagination_id(timestamp, last_data=[], data=[]):
    # Like binance, bybit pagination feels like an IQ test
    if len(data):
        last_trade = data[-1]
        pagination_id = last_trade["id"] - len(data)
        assert pagination_id > 0
        return pagination_id


def get_bybit_api_timestamp(trade):
    return parse_datetime(trade["time"])


def get_trades(symbol, timestamp_from, pagination_id, log_format=None):
    url = f"{API_URL}/trading-records?symbol={symbol}&limit={MAX_RESULTS}"
    return iter_api(
        url,
        get_bybit_api_pagination_id,
        get_bybit_api_timestamp,
        get_bybit_api_response,
        MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=pagination_id,
        log_format=log_format,
    )


def get_bybit_api_response(url, pagination_id=None, retry=30):
    throttle_api_requests(
        BYBIT_MAX_REQUESTS_RESET,
        BYBIT_TOTAL_REQUESTS,
        MAX_REQUESTS_RESET,
        MAX_REQUESTS,
    )
    try:
        response = httpx.get(get_bybit_api_url(url, pagination_id))
        increment_api_total_requests(BYBIT_TOTAL_REQUESTS)
        if response.status_code == 200:
            result = response.read()
            data = json.loads(result, parse_float=Decimal)
            assert data["ret_msg"] == "OK"
            res = data["result"]
            # If no pagination_id, ascending order
            if pagination_id:
                # Descending order, please
                res.reverse()
            return res
        else:
            response.raise_for_status()
    except HTTPX_ERRORS:
        if retry > 0:
            time.sleep(1)
            retry -= 1
            return get_bybit_api_response(url, pagination_id, retry)
        raise
