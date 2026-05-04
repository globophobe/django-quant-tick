from datetime import datetime
from decimal import Decimal, InvalidOperation

import pandas as pd
from pandas import DataFrame

from quant_tick.exchanges.funding import ExchangeFunding

from .api import get_binance_api_response
from .constants import FUTURES_API_URL

BINANCE_FUNDING_MAX_RESULTS = 1000


class BinanceFuturesFunding(ExchangeFunding):
    interval = pd.Timedelta("8h")
    timestamp_anomaly_tolerance = pd.Timedelta("1min")


def parse_optional_decimal(value: object) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


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
        return BinanceFuturesFunding.empty_frame(["funding_rate", "mark_price"])

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
        return BinanceFuturesFunding.empty_frame(["funding_rate", "mark_price"])

    df = DataFrame(
        {
            "timestamp": pd.to_datetime(
                [int(item["fundingTime"]) for item in rows],
                unit="ms",
                utc=True,
            ),
            "funding_rate": [Decimal(str(item["fundingRate"])) for item in rows],
            "mark_price": [
                parse_optional_decimal(item.get("markPrice"))
                for item in rows
            ],
        }
    )
    return BinanceFuturesFunding.normalize_frame(df, timestamp_from, timestamp_to)
