from datetime import datetime
from decimal import Decimal
from urllib.parse import urlencode

import pandas as pd
from pandas import DataFrame

from quant_tick.exchanges.funding import ExchangeFunding

from .api import get_coinbase_advanced_api_response
from .constants import FUNDING_MAX_RESULTS, INTX_API_URL


class CoinbaseAdvancedFunding(ExchangeFunding):
    interval = pd.Timedelta("1h")
    timestamp_anomaly_tolerance = pd.Timedelta("5min")


def normalize_coinbase_advanced_funding_symbol(api_symbol: str) -> str:
    return str(api_symbol).strip().removesuffix("-INTX")


def get_coinbase_advanced_funding_url(api_symbol: str, offset: int) -> str:
    query = urlencode(
        {
            "result_limit": FUNDING_MAX_RESULTS,
            "result_offset": offset,
        }
    )
    symbol = normalize_coinbase_advanced_funding_symbol(api_symbol)
    return f"{INTX_API_URL}/instruments/{symbol}/funding?{query}"


def get_coinbase_advanced_funding_response(
    api_symbol: str,
    offset: int = 0,
) -> list[dict]:
    response = get_coinbase_advanced_api_response(
        get_coinbase_advanced_funding_url(api_symbol, offset)
    )
    return response.get("results", [])


def coinbase_advanced_funding(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
) -> DataFrame:
    if timestamp_to <= timestamp_from:
        return CoinbaseAdvancedFunding.empty_frame(["funding_rate", "mark_price"])

    from_ts = pd.to_datetime(timestamp_from, utc=True)
    offset = 0
    rows = []
    while True:
        data = get_coinbase_advanced_funding_response(api_symbol, offset)
        if not data:
            break
        rows.extend(data)
        timestamps = pd.to_datetime(
            [item["event_time"] for item in data],
            format="ISO8601",
            utc=True,
        )
        if timestamps.min() < from_ts:
            break
        offset += len(data)
        if len(data) < FUNDING_MAX_RESULTS:
            break

    if not rows:
        return CoinbaseAdvancedFunding.empty_frame(["funding_rate", "mark_price"])

    df = DataFrame(
        {
            "timestamp": pd.to_datetime(
                [item["event_time"] for item in rows],
                format="ISO8601",
                utc=True,
            ),
            "funding_rate": [Decimal(str(item["funding_rate"])) for item in rows],
            "mark_price": [
                Decimal(str(item["mark_price"]))
                for item in rows
            ],
        }
    )
    return CoinbaseAdvancedFunding.normalize_frame(df, timestamp_from, timestamp_to)
