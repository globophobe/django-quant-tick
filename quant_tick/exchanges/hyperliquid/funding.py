from datetime import datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.exchanges.funding import ExchangeFunding

from .api import normalize_coin, post_hyperliquid_info, to_millis
from .constants import FUNDING_INTERVAL_MS, FUNDING_MAX_RESULTS


class HyperliquidFunding(ExchangeFunding):
    interval = pd.Timedelta("1h")
    timestamp_anomaly_tolerance = pd.Timedelta("1min")


def get_hyperliquid_funding_response(
    api_symbol: str,
    start_ms: int,
    end_ms: int,
) -> list[dict]:
    payload = {
        "type": "fundingHistory",
        "coin": normalize_coin(api_symbol),
        "startTime": start_ms,
        "endTime": end_ms,
    }
    return post_hyperliquid_info(payload)


def hyperliquid_funding(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
) -> DataFrame:
    """Fetch Hyperliquid perpetual funding events."""
    start_ms = to_millis(timestamp_from)
    end_ms = to_millis(timestamp_to)
    if end_ms <= start_ms:
        return HyperliquidFunding.empty_frame(["funding_rate", "premium"])

    cursor = start_ms
    rows = []
    while cursor < end_ms:
        data = get_hyperliquid_funding_response(api_symbol, cursor, end_ms)
        if not data:
            break
        rows.extend(data)
        last_time = max(int(item["time"]) for item in data)
        next_cursor = max(cursor + 1, last_time + 1)
        if next_cursor <= cursor:
            cursor = min(end_ms, cursor + FUNDING_INTERVAL_MS)
        else:
            cursor = next_cursor
        if len(data) < FUNDING_MAX_RESULTS:
            break

    if not rows:
        return HyperliquidFunding.empty_frame(["funding_rate", "premium"])

    df = DataFrame(
        {
            "timestamp": pd.to_datetime(
                [int(item["time"]) for item in rows],
                unit="ms",
                utc=True,
            ),
            "funding_rate": [Decimal(str(item["fundingRate"])) for item in rows],
            "premium": [Decimal(str(item["premium"])) for item in rows],
        }
    )
    return HyperliquidFunding.normalize_frame(df, timestamp_from, timestamp_to)
