from datetime import datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from .api import get_binance_api_response
from .constants import FUTURES_API_URL

BINANCE_FUNDING_MAX_RESULTS = 1000


def format_binance_funding_timestamp(timestamp: datetime) -> int:
    return int(timestamp.timestamp() * 1000)


def get_binance_funding_url(
    url: str,
    timestamp_from: datetime | None = None,
    pagination_id: int | None = None,
) -> str:
    return url


def get_binance_funding_response(base_url: str) -> list[dict]:
    return get_binance_api_response(
        get_binance_funding_url,
        base_url,
        reverse=False,
    )


def binance_funding(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
) -> DataFrame:
    """Fetch Binance USD-M perpetual funding events."""
    if timestamp_to <= timestamp_from:
        empty = DataFrame(columns=["timestamp", "funding_rate", "mark_price"])
        return empty.set_index("timestamp")

    cursor = timestamp_from
    rows = []
    while cursor < timestamp_to:
        url = (
            f"{FUTURES_API_URL}/fundingRate"
            f"?symbol={str(api_symbol).strip()}"
            f"&startTime={format_binance_funding_timestamp(cursor)}"
            f"&endTime={format_binance_funding_timestamp(timestamp_to)}"
            f"&limit={BINANCE_FUNDING_MAX_RESULTS}"
        )
        data = get_binance_funding_response(url)
        if not data:
            break
        rows.extend(data)
        last_time = max(int(item["fundingTime"]) for item in data)
        next_cursor = pd.to_datetime(last_time + 1, unit="ms", utc=True).to_pydatetime()
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(data) < BINANCE_FUNDING_MAX_RESULTS:
            break

    if not rows:
        empty = DataFrame(columns=["timestamp", "funding_rate", "mark_price"])
        return empty.set_index("timestamp")

    from_ts = pd.to_datetime(timestamp_from, utc=True)
    to_ts = pd.to_datetime(timestamp_to, utc=True)
    df = DataFrame(
        {
            "timestamp": pd.to_datetime(
                [int(item["fundingTime"]) for item in rows],
                unit="ms",
                utc=True,
            ),
            "funding_rate": [Decimal(str(item["fundingRate"])) for item in rows],
            "mark_price": [
                None
                if item.get("markPrice") in {None, ""}
                else Decimal(str(item["markPrice"]))
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
