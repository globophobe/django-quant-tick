import datetime
import json
import re
import time
from decimal import Decimal

import httpx

from cryptofeed_werks.controllers import HTTPX_ERRORS, iter_api
from cryptofeed_werks.lib import parse_datetime

from .constants import API_URL, MAX_RESULTS, MIN_ELAPSED_PER_REQUEST, MONTHS


def get_bitmex_api_url(url, pagination_id):
    url += f"&count={MAX_RESULTS}&reverse=true"
    if pagination_id:
        return url + f"&endTime={pagination_id}"
    return url


def get_bitmex_api_pagination_id(timestamp, last_data=[], data=[]):
    return format_bitmex_api_timestamp(timestamp)


def get_bitmex_api_timestamp(trade):
    return parse_datetime(trade["timestamp"])


def format_bitmex_api_timestamp(timestamp):
    return timestamp.replace(tzinfo=None).isoformat()


def get_active_futures(root_symbol, timestamp_from, pagination_id, log_format=None):
    endpoint = "instrument/active"
    return get_futures(
        endpoint, root_symbol, timestamp_from, pagination_id, log_format=log_format
    )


def get_expired_futures(root_symbol, timestamp_from, pagination_id, log_format=None):
    endpoint = "instrument"
    return get_futures(
        endpoint, root_symbol, timestamp_from, pagination_id, log_format=log_format
    )


def get_futures(endpoint, root_symbol, timestamp_from, pagination_id, log_format=None):
    filters = json.dumps({"rootSymbol": root_symbol})
    url = f"{API_URL}/{endpoint}?filter={filters}"
    timestamp_key = "timestamp"
    results = iter_api(
        url,
        timestamp_key,
        get_bitmex_api_pagination_id,
        get_bitmex_api_response,
        MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=pagination_id,
        log_format=log_format,
    )
    instruments = []
    regex = re.compile(f"^{root_symbol}" + r"(\w)\d+$")
    for instrument in results:
        symbol = instrument["symbol"]
        match = regex.match(symbol)
        if match:
            is_future = match.group(1) in MONTHS
            if is_future:
                listing = parse_datetime(instrument["listing"])
                expiry = parse_datetime(instrument["expiry"])
                if expiry >= timestamp_from:
                    instruments.append(
                        {"symbol": symbol, "listing": listing, "expiry": expiry}
                    )
    return instruments


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


def get_trades(symbol, timestamp_from, pagination_id, log_format=None):
    url = f"{API_URL}/trade?symbol={symbol}"
    return iter_api(
        url,
        get_bitmex_api_pagination_id,
        get_bitmex_api_timestamp,
        get_bitmex_api_response,
        MAX_RESULTS,
        MIN_ELAPSED_PER_REQUEST,
        timestamp_from=timestamp_from,
        pagination_id=pagination_id,
        log_format=log_format,
    )


def get_bitmex_api_response(url, pagination_id=None, retry=30):
    try:
        response = httpx.get(get_bitmex_api_url(url, pagination_id))
        if response.status_code == 200:
            remaining = response.headers["x-ratelimit-remaining"]
            if remaining == 0:
                timestamp = datetime.datetime.utcnow().timestamp()
                reset = response.headers["x-ratelimit-reset"]
                if reset > timestamp:
                    sleep_duration = reset - timestamp
                    print(f"Max requests, sleeping {sleep_duration} seconds")
                    time.sleep(sleep_duration)
            result = response.read()
            return json.loads(result, parse_float=Decimal)
        elif response.status_code == 429:
            retry = response.headers.get("Retry-After", 1)
            time.sleep(int(retry))
        else:
            response.raise_for_status()
    except HTTPX_ERRORS:
        if retry > 0:
            time.sleep(1)
            retry -= 1
            return get_bitmex_api_response(url, pagination_id, retry)
        raise
