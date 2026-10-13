from datetime import timedelta
from decimal import Decimal

from quant_tick.lib import (
    candles_to_data_frame,
    parse_datetime,
    parse_fixed_resolution_minutes,
    resample_candles,
)

from .api import get_upbit_response


def upbit_candles(api_symbol, timestamp_from, timestamp_to, resolution="1m"):
    """Retain quote-currency prices and turnover, aggregating traded intervals.

    Upbit omits zero-trade intervals, so elapsed resampled windows can contain
    fewer source candles than their duration would otherwise require.
    """
    target = parse_fixed_resolution_minutes(resolution)
    source = max(
        value for value in (1, 3, 5, 10, 15, 30, 60, 240) if target % value == 0
    )
    cursor = timestamp_to
    rows = []
    while True:
        page = get_upbit_response(
            f"/candles/minutes/{source}",
            {"market": api_symbol, "to": cursor.isoformat(), "count": 200},
        )
        if not page:
            break
        for row in page:
            timestamp = parse_datetime(row["candle_date_time_utc"] + "+00:00")
            if timestamp >= cursor:
                raise ValueError("Upbit candle pagination did not advance")
            cursor = timestamp
            if timestamp + timedelta(minutes=source) > timestamp_to:
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    **{
                        name: Decimal(str(row[key]))
                        for name, key in (
                            ("open", "opening_price"),
                            ("high", "high_price"),
                            ("low", "low_price"),
                            ("close", "trade_price"),
                            ("notional", "candle_acc_trade_volume"),
                            ("volume", "candle_acc_trade_price"),
                        )
                    },
                }
            )
        if cursor <= timestamp_from:
            break
    frame = candles_to_data_frame(timestamp_from, timestamp_to, rows)
    if source == target:
        return frame
    return resample_candles(
        frame,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        resolution_minutes=target,
        source_resolution_minutes=source,
        missing_candles_are_zero=True,
    )
