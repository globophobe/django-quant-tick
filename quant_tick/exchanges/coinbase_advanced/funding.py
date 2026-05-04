from datetime import datetime
from decimal import Decimal
from urllib.parse import urlencode

import pandas as pd
from pandas import DataFrame

from .api import get_coinbase_advanced_api_response
from .constants import FUNDING_MAX_RESULTS, INTX_API_URL


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
        empty = DataFrame(columns=["timestamp", "funding_rate", "mark_price"])
        return empty.set_index("timestamp")

    from_ts = pd.to_datetime(timestamp_from, utc=True)
    to_ts = pd.to_datetime(timestamp_to, utc=True)
    offset = 0
    rows = []
    while True:
        data = get_coinbase_advanced_funding_response(api_symbol, offset)
        if not data:
            break
        rows.extend(data)
        timestamps = pd.to_datetime(
            [item["event_time"] for item in data],
            utc=True,
        )
        if timestamps.min() < from_ts:
            break
        offset += len(data)
        if len(data) < FUNDING_MAX_RESULTS:
            break

    if not rows:
        empty = DataFrame(columns=["timestamp", "funding_rate", "mark_price"])
        return empty.set_index("timestamp")

    df = DataFrame(
        {
            "timestamp": pd.to_datetime(
                [item["event_time"] for item in rows],
                utc=True,
            ),
            "funding_rate": [Decimal(str(item["funding_rate"])) for item in rows],
            "mark_price": [
                Decimal(str(item["mark_price"]))
                for item in rows
            ],
        }
    )
    df = (
        df.sort_values("timestamp", kind="stable")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .loc[lambda frame: (frame["timestamp"] >= from_ts) & (frame["timestamp"] < to_ts)]
    )
    return df.set_index("timestamp")
