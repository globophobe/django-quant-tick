from datetime import datetime
from decimal import Decimal, InvalidOperation

import pandas as pd
from pandas import DataFrame

from quant_tick.exchanges.funding import ExchangeFunding

from quant_tick.exchanges.binance.api import get_binance_api_response

from .constants import API_URL
from .market_history import binance_market_history, empty_market_history

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


def binance_futures_funding(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    *,
    funding_interval: str | pd.Timedelta | None = None,
) -> DataFrame:
    """Fetch Binance USD-M funding with aligned futures market history."""
    columns = ["funding_rate", "mark_price", *empty_market_history().columns]
    if timestamp_to <= timestamp_from:
        return BinanceFuturesFunding.empty_frame(columns)

    cursor = timestamp_from
    rows = []
    while cursor < timestamp_to:
        url = (
            f"{API_URL}/fundingRate"
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
        return BinanceFuturesFunding.empty_frame(columns)

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
    normalized = BinanceFuturesFunding.normalize_frame(
        df,
        timestamp_from,
        timestamp_to,
        interval=funding_interval,
    )
    history = binance_market_history(api_symbol, timestamp_from, timestamp_to)
    return normalized.join(history, how="left").sort_index(kind="stable")
