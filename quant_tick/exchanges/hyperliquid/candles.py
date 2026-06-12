from datetime import datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.lib import candles_to_data_frame, parse_fixed_resolution_minutes

from .api import normalize_coin, post_hyperliquid_info, to_millis
from .constants import CANDLE_MAX_RESULTS, INTERVAL_MILLIS, INTERVALS


def get_hyperliquid_interval(resolution: str | int | None) -> str:
    if resolution is None:
        return "1m"
    raw = str(resolution).strip()
    if raw in INTERVALS:
        return INTERVALS[raw]
    minutes = parse_fixed_resolution_minutes(resolution)
    interval = f"{minutes}m"
    if interval in INTERVALS:
        return INTERVALS[interval]
    if minutes % 1440 == 0:
        interval = f"{minutes // 1440}d"
    elif minutes % 60 == 0:
        interval = f"{minutes // 60}h"
    if interval in INTERVALS:
        return INTERVALS[interval]
    raise ValueError(f"Unsupported Hyperliquid resolution: {resolution}")


def get_hyperliquid_frequency(interval: str) -> int:
    interval = get_hyperliquid_interval(interval)
    return int(INTERVAL_MILLIS[interval] // 60_000)


def get_hyperliquid_candle_response(
    api_symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[dict]:
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": normalize_coin(api_symbol),
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }
    return post_hyperliquid_info(payload)


def hyperliquid_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    resolution: str | int | None = "1m",
) -> DataFrame:
    """Fetch Hyperliquid direct venue candles."""
    interval = get_hyperliquid_interval(resolution)
    interval_ms = INTERVAL_MILLIS[interval]
    start_ms = to_millis(timestamp_from)
    end_ms = to_millis(timestamp_to)
    if end_ms <= start_ms:
        empty = DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "notional"]
        )
        return empty.set_index("timestamp")

    cursor = start_ms
    chunk_ms = interval_ms * CANDLE_MAX_RESULTS
    rows = []
    while cursor < end_ms:
        chunk_end = min(cursor + chunk_ms, end_ms)
        data = get_hyperliquid_candle_response(
            api_symbol,
            interval,
            cursor,
            chunk_end,
        )
        if not data:
            cursor = chunk_end
            continue
        rows.extend(data)
        last_open_ms = max(int(item["t"]) for item in data)
        next_cursor = max(cursor + interval_ms, last_open_ms + interval_ms)
        cursor = chunk_end if next_cursor <= cursor else next_cursor

    if not rows:
        empty = DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "notional"]
        )
        return empty.set_index("timestamp")

    df = DataFrame(
        {
            "timestamp": pd.to_datetime(
                [int(item["t"]) for item in rows],
                unit="ms",
                utc=True,
            ),
            "open": [Decimal(str(item["o"])) for item in rows],
            "high": [Decimal(str(item["h"])) for item in rows],
            "low": [Decimal(str(item["l"])) for item in rows],
            "close": [Decimal(str(item["c"])) for item in rows],
            "notional": [Decimal(str(item["v"])) for item in rows],
            "trades": [
                None if item.get("n") is None else int(item["n"])
                for item in rows
            ],
        }
    )
    return candles_to_data_frame(
        timestamp_from,
        timestamp_to,
        df.to_dict("records"),
        reverse=False,
    )
