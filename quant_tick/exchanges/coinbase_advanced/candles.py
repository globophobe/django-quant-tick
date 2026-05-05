import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from urllib.parse import urlencode

import pandas as pd
from pandas import DataFrame

from quant_tick.lib import (
    candles_to_data_frame,
    parse_fixed_resolution_minutes,
    resample_candles,
)

from .api import get_coinbase_advanced_api_response
from .constants import CANDLE_MAX_RESULTS, MARKET_API_URL

logger = logging.getLogger(__name__)

COINBASE_ADVANCED_GRANULARITY_BY_MINUTES = {
    1: "ONE_MINUTE",
    5: "FIVE_MINUTE",
    15: "FIFTEEN_MINUTE",
    30: "THIRTY_MINUTE",
    60: "ONE_HOUR",
    120: "TWO_HOUR",
    240: "FOUR_HOUR",
    360: "SIX_HOUR",
    1440: "ONE_DAY",
}


def to_unix_seconds(timestamp: datetime) -> int:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    else:
        ts = ts.tz_convert(UTC)
    return int(ts.timestamp())


def get_coinbase_advanced_fetch_granularity(
    resolution: str | int | None,
) -> tuple[int, int, str]:
    target_minutes = parse_fixed_resolution_minutes(resolution)
    if target_minutes in COINBASE_ADVANCED_GRANULARITY_BY_MINUTES:
        return (
            target_minutes,
            target_minutes,
            COINBASE_ADVANCED_GRANULARITY_BY_MINUTES[target_minutes],
        )
    for minutes in sorted(COINBASE_ADVANCED_GRANULARITY_BY_MINUTES, reverse=True):
        if target_minutes % minutes == 0:
            return (
                target_minutes,
                minutes,
                COINBASE_ADVANCED_GRANULARITY_BY_MINUTES[minutes],
            )
    return target_minutes, 1, COINBASE_ADVANCED_GRANULARITY_BY_MINUTES[1]


def get_coinbase_advanced_candle_url(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    granularity: str,
) -> str:
    query = urlencode(
        {
            "start": to_unix_seconds(timestamp_from),
            "end": to_unix_seconds(timestamp_to),
            "granularity": granularity,
            "limit": CANDLE_MAX_RESULTS,
        }
    )
    return f"{MARKET_API_URL}/products/{api_symbol}/candles?{query}"


def get_coinbase_advanced_candle_response(url: str) -> list[dict]:
    response = get_coinbase_advanced_api_response(url)
    return response.get("candles", [])


def get_coinbase_advanced_candle_timestamp(candle: dict) -> datetime:
    return datetime.fromtimestamp(int(candle["start"]), tz=UTC)


def fetch_coinbase_advanced_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    *,
    granularity_minutes: int,
    granularity: str,
    log_format: str | None = None,
) -> DataFrame:
    if timestamp_to <= timestamp_from:
        empty = DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "notional"]
        )
        return empty.set_index("timestamp")

    step = timedelta(minutes=granularity_minutes)
    chunk = step * CANDLE_MAX_RESULTS
    cursor_to = timestamp_to
    rows = []
    while cursor_to > timestamp_from:
        cursor_from = max(timestamp_from, cursor_to - chunk)
        inclusive_to = cursor_to - step
        if inclusive_to < cursor_from:
            break
        url = get_coinbase_advanced_candle_url(
            api_symbol,
            cursor_from,
            inclusive_to,
            granularity,
        )
        data = get_coinbase_advanced_candle_response(url)
        rows.extend(data)
        if data and log_format:
            timestamp = min(get_coinbase_advanced_candle_timestamp(row) for row in data)
            logger.info(log_format.format(timestamp=timestamp.isoformat()))
        cursor_to = cursor_from

    if not rows:
        empty = DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "notional"]
        )
        return empty.set_index("timestamp")

    candles = [
        {
            "timestamp": get_coinbase_advanced_candle_timestamp(row),
            "open": Decimal(str(row["open"])),
            "high": Decimal(str(row["high"])),
            "low": Decimal(str(row["low"])),
            "close": Decimal(str(row["close"])),
            "notional": Decimal(str(row["volume"])),
        }
        for row in rows
    ]
    candles = sorted(candles, key=lambda row: row["timestamp"])
    return candles_to_data_frame(
        timestamp_from,
        timestamp_to,
        candles,
        reverse=False,
    )


def coinbase_advanced_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    resolution: str | int | None = "1m",
    log_format: str | None = None,
) -> DataFrame:
    target_minutes, fetch_minutes, granularity = (
        get_coinbase_advanced_fetch_granularity(resolution)
    )
    df = fetch_coinbase_advanced_candles(
        api_symbol,
        timestamp_from,
        timestamp_to,
        granularity_minutes=fetch_minutes,
        granularity=granularity,
        log_format=log_format,
    )
    if fetch_minutes == target_minutes:
        return df
    return resample_candles(
        df,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        resolution_minutes=target_minutes,
        source_resolution_minutes=fetch_minutes,
    )
