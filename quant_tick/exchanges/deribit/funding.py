from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.exchanges.funding import ExchangeFunding
from quant_tick.lib import to_decimal_or_none

from .api import get_deribit_result, to_millis
from .constants import FUNDING_FETCH_WINDOW

FUNDING_HISTORY_START = datetime(2019, 4, 30, 10, tzinfo=UTC)

KNOWN_MISSING_FUNDING_TIMESTAMPS = frozenset(
    {
        datetime(2020, 8, 27, 6, tzinfo=UTC),
        datetime(2020, 8, 27, 7, tzinfo=UTC),
    }
)


class DeribitFunding(ExchangeFunding):
    interval = pd.Timedelta("1h")
    timestamp_anomaly_tolerance = pd.Timedelta("1min")

    @classmethod
    def is_known_missing_timestamp(cls, timestamp: datetime) -> bool:
        return (
            timestamp < FUNDING_HISTORY_START
            or timestamp in KNOWN_MISSING_FUNDING_TIMESTAMPS
        )


def get_deribit_funding_response(
    api_symbol: str,
    start_ms: int,
    end_ms: int,
) -> list[dict]:
    return get_deribit_result(
        "get_funding_rate_history",
        {
            "instrument_name": str(api_symbol).strip(),
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
        },
    )


def deribit_funding(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
) -> DataFrame:
    """Fetch Deribit funding."""
    start_ms = to_millis(timestamp_from)
    end_ms = to_millis(timestamp_to)
    columns = [
        "funding_rate",
        "interest_8h",
        "index_price",
        "prev_index_price",
    ]
    if end_ms <= start_ms:
        return DeribitFunding.empty_frame(columns)

    window_ms = int(FUNDING_FETCH_WINDOW.total_seconds() * 1000)
    cursor = start_ms
    inclusive_end_ms = end_ms - 1
    rows = []
    while cursor <= inclusive_end_ms:
        chunk_end = min(cursor + window_ms - 1, inclusive_end_ms)
        # Deribit excludes an event exactly equal to start_timestamp.
        rows.extend(
            get_deribit_funding_response(api_symbol, max(0, cursor - 1), chunk_end)
        )
        cursor = chunk_end + 1

    if not rows:
        return DeribitFunding.empty_frame(columns)

    df = DataFrame(
        {
            "timestamp": pd.to_datetime(
                [int(item["timestamp"]) for item in rows],
                unit="ms",
                utc=True,
            ),
            "funding_rate": [
                Decimal(str(item["interest_1h"])) for item in rows
            ],
            "interest_8h": [
                Decimal(str(item["interest_8h"])) for item in rows
            ],
            "index_price": [
                Decimal(str(item["index_price"])) for item in rows
            ],
            "prev_index_price": [
                to_decimal_or_none(item.get("prev_index_price")) for item in rows
            ],
        }
    )
    return DeribitFunding.normalize_frame(df, timestamp_from, timestamp_to)
