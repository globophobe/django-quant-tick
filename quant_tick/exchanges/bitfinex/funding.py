from datetime import datetime
from decimal import Decimal, InvalidOperation

import pandas as pd
from pandas import DataFrame

from quant_tick.exchanges.funding import ExchangeFunding

from .api import format_bitfinex_api_timestamp, get_bitfinex_api_response
from .constants import API_URL, FUNDING_MAX_RESULTS

FUNDING_LOOKBACK = pd.Timedelta("1h")
STATUS_MTS = 0
NEXT_FUNDING_EVT_MTS = 7
NEXT_FUNDING_ACCRUED = 8
NEXT_FUNDING_STEP = 9
CURRENT_FUNDING = 11
MARK_PRICE = 14
OPEN_INTEREST = 17
FUNDING_CLAMP_MIN = 21
FUNDING_CLAMP_MAX = 22


class BitfinexFunding(ExchangeFunding):
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


def get_status_value(row: list, index: int) -> object | None:
    if len(row) <= index:
        return None
    return row[index]


def get_bitfinex_funding_url(
    url: str,
    timestamp_from: datetime | None = None,
    pagination_id: int | None = None,
) -> str:
    return url


def get_bitfinex_funding_response(base_url: str) -> list[list]:
    return get_bitfinex_api_response(get_bitfinex_funding_url, base_url)


def get_bitfinex_funding_rows(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
) -> list[list]:
    from_ts = pd.to_datetime(timestamp_from, utc=True)
    fetch_from = (from_ts - FUNDING_LOOKBACK).to_pydatetime()
    cursor = pd.to_datetime(timestamp_to, utc=True).to_pydatetime()
    rows = []
    while cursor > fetch_from:
        url = (
            f"{API_URL}/status/deriv/{str(api_symbol).strip()}/hist"
            f"?limit={FUNDING_MAX_RESULTS}"
            "&sort=-1"
            f"&start={format_bitfinex_api_timestamp(fetch_from)}"
            f"&end={format_bitfinex_api_timestamp(cursor)}"
        )
        data = get_bitfinex_funding_response(url)
        if not data:
            break
        rows.extend(data)
        timestamps = pd.to_datetime(
            [int(get_status_value(item, STATUS_MTS)) for item in data],
            unit="ms",
            utc=True,
        )
        oldest = timestamps.min()
        if oldest <= pd.Timestamp(fetch_from):
            break
        next_cursor = (oldest - pd.Timedelta("1ms")).to_pydatetime()
        if next_cursor >= cursor:
            break
        cursor = next_cursor
        if len(data) < FUNDING_MAX_RESULTS:
            break
    return rows


def get_bitfinex_event_rows(rows: list[list]) -> list[dict]:
    event_rows = []
    for row in rows:
        status_ms = get_status_value(row, STATUS_MTS)
        event_ms = get_status_value(row, NEXT_FUNDING_EVT_MTS)
        funding_rate = parse_optional_decimal(
            get_status_value(row, NEXT_FUNDING_ACCRUED)
        )
        if status_ms is None or event_ms is None or funding_rate is None:
            continue
        status_timestamp = pd.to_datetime(int(status_ms), unit="ms", utc=True)
        event_timestamp = pd.to_datetime(int(event_ms), unit="ms", utc=True)
        if status_timestamp >= event_timestamp:
            continue
        event_rows.append(
            {
                "timestamp": event_timestamp,
                "funding_rate": funding_rate,
                "status_timestamp": status_timestamp,
                "next_funding_step": get_status_value(row, NEXT_FUNDING_STEP),
                "current_funding": parse_optional_decimal(
                    get_status_value(row, CURRENT_FUNDING)
                ),
                "mark_price": parse_optional_decimal(get_status_value(row, MARK_PRICE)),
                "open_interest": parse_optional_decimal(
                    get_status_value(row, OPEN_INTEREST)
                ),
                "funding_clamp_min": parse_optional_decimal(
                    get_status_value(row, FUNDING_CLAMP_MIN)
                ),
                "funding_clamp_max": parse_optional_decimal(
                    get_status_value(row, FUNDING_CLAMP_MAX)
                ),
            }
        )
    return event_rows


def bitfinex_funding(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
) -> DataFrame:
    """Fetch Bitfinex perpetual funding events from derivatives status history."""
    columns = [
        "funding_rate",
        "status_timestamp",
        "next_funding_step",
        "current_funding",
        "mark_price",
        "open_interest",
        "funding_clamp_min",
        "funding_clamp_max",
    ]
    if timestamp_to <= timestamp_from:
        return BitfinexFunding.empty_frame(columns)

    rows = get_bitfinex_funding_rows(api_symbol, timestamp_from, timestamp_to)
    event_rows = get_bitfinex_event_rows(rows)
    if not event_rows:
        return BitfinexFunding.empty_frame(columns)

    df = DataFrame(event_rows)
    df = df.sort_values(["timestamp", "status_timestamp"], kind="stable")
    # Bitfinex status history can include multiple status rows for the same
    # funding event. The latest status before the event is the canonical row.
    df = df.groupby("timestamp", sort=False, as_index=False).tail(1)
    return BitfinexFunding.normalize_frame(df, timestamp_from, timestamp_to)
