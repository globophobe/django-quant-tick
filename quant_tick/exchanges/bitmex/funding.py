from datetime import datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.exchanges.funding import ExchangeFunding

from .api import get_bitmex_api_response
from .constants import API_URL

BITMEX_FUNDING_MAX_RESULTS = 500


class BitmexFunding(ExchangeFunding):
    interval = pd.Timedelta("8h")
    anchor_offset = pd.Timedelta("4h")
    timestamp_anomaly_tolerance = pd.Timedelta("1min")


def format_bitmex_funding_timestamp(timestamp: datetime) -> str:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def get_bitmex_funding_url(
    url: str,
    timestamp_from: datetime | None = None,
    pagination_id: str | None = None,
) -> str:
    return url


def get_bitmex_funding_response(base_url: str) -> list[dict]:
    return get_bitmex_api_response(get_bitmex_funding_url, base_url)


def bitmex_funding(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
) -> DataFrame:
    """Fetch BitMEX perpetual funding events."""
    if timestamp_to <= timestamp_from:
        return BitmexFunding.empty_frame(["funding_rate"])

    cursor = timestamp_to
    from_ts = pd.Timestamp(timestamp_from)
    rows = []
    while cursor > timestamp_from:
        url = (
            f"{API_URL}/funding"
            f"?symbol={str(api_symbol).strip()}"
            f"&count={BITMEX_FUNDING_MAX_RESULTS}"
            "&reverse=true"
            f"&endTime={format_bitmex_funding_timestamp(cursor)}"
        )
        data = get_bitmex_funding_response(url)
        if not data:
            break
        rows.extend(data)
        timestamps = pd.to_datetime(
            [item["timestamp"] for item in data],
            utc=True,
            format="ISO8601",
        )
        oldest = timestamps.min()
        if oldest <= from_ts:
            break
        next_cursor = (oldest - pd.Timedelta("1ms")).to_pydatetime()
        if next_cursor >= cursor:
            break
        cursor = next_cursor
        if len(data) < BITMEX_FUNDING_MAX_RESULTS:
            break

    if not rows:
        return BitmexFunding.empty_frame(["funding_rate"])

    frame_dict = {
        "timestamp": pd.to_datetime([item["timestamp"] for item in rows], utc=True),
        "funding_rate": [Decimal(str(item["fundingRate"])) for item in rows],
    }
    if any("fundingRateDaily" in item for item in rows):
        frame_dict["funding_rate_daily"] = [
            None
            if item.get("fundingRateDaily") is None
            else Decimal(str(item["fundingRateDaily"]))
            for item in rows
        ]
    if any("indicativeFundingRate" in item for item in rows):
        frame_dict["indicative_funding_rate"] = [
            None
            if item.get("indicativeFundingRate") is None
            else Decimal(str(item["indicativeFundingRate"]))
            for item in rows
        ]
    df = DataFrame(frame_dict).sort_values("timestamp", kind="stable")
    return BitmexFunding.normalize_frame(df, timestamp_from, timestamp_to)
