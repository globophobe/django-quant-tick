from datetime import timedelta
from decimal import Decimal

from quant_tick.lib import (
    candles_to_data_frame,
    parse_datetime,
    parse_fixed_resolution_minutes,
    resample_candles,
)

from .api import get_okx_response

RESOLUTIONS = {
    1: "1m",
    3: "3m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "1H",
    120: "2H",
    240: "4H",
    360: "6Hutc",
    720: "12Hutc",
    1440: "1Dutc",
}


def okx_candles(api_symbol, timestamp_from, timestamp_to, resolution="1m"):
    target = parse_fixed_resolution_minutes(resolution)
    source = max(minutes for minutes in RESOLUTIONS if target % minutes == 0)
    cursor = int(timestamp_to.timestamp() * 1000)
    rows = []
    while True:
        page = get_okx_response(
            "/market/history-candles",
            {
                "instId": api_symbol,
                "bar": RESOLUTIONS[source],
                "after": cursor,
                "before": int(timestamp_from.timestamp() * 1000) - 1,
                "limit": 100,
            },
        )
        if not page:
            break
        previous = cursor
        for row in page:
            timestamp_ms = int(row[0])
            if timestamp_ms >= previous:
                raise ValueError("OKX candle pagination did not advance")
            previous = timestamp_ms
            timestamp = parse_datetime(timestamp_ms, unit="ms")
            if row[8] != "1" or timestamp + timedelta(minutes=source) > timestamp_to:
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    **{
                        name: Decimal(row[i])
                        for i, name in enumerate(("open", "high", "low", "close"), 1)
                    },
                    "notional": Decimal(
                        row[6] if api_symbol.endswith("-SWAP") else row[5]
                    ),
                    "volume": Decimal(row[7]),
                }
            )
        cursor = previous
        if parse_datetime(cursor, unit="ms") <= timestamp_from:
            break
    frame = candles_to_data_frame(timestamp_from, timestamp_to, rows)
    if target == source:
        return frame
    return resample_candles(
        frame,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        resolution_minutes=target,
        source_resolution_minutes=source,
    )
