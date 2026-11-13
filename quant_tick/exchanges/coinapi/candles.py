"""CoinAPI OHLCV reports base quantity, not quote turnover.

Keep missing validation candles indeterminate: both datasets come from CoinAPI,
so agreement establishes ingestion consistency, not independent venue coverage.
"""

from datetime import timedelta
from decimal import Decimal

from quant_tick.lib import (
    candles_to_data_frame,
    parse_datetime,
    parse_fixed_resolution_minutes,
    resample_candles,
)

from .api import get_coinapi_response, require_spot_symbol

PAGE_SIZE = 1000


def coinapi_candles(api_symbol, timestamp_from, timestamp_to, resolution="1m"):
    require_spot_symbol(api_symbol)
    target = parse_fixed_resolution_minutes(resolution)
    # Hour and day sources avoid downloading minute bars for larger requests.
    source = next(value for value in (1440, 60, 1) if target % value == 0)
    period = {1440: "1DAY", 60: "1HRS", 1: "1MIN"}[source]
    interval = timedelta(minutes=source)
    cursor = timestamp_from
    rows = []
    while cursor < timestamp_to:
        page = get_coinapi_response(
            f"/v1/ohlcv/{api_symbol}/history",
            {
                "period_id": period,
                "time_start": cursor.isoformat(),
                "time_end": timestamp_to.isoformat(),
                "limit": PAGE_SIZE,
            },
        )
        previous = None
        for row in page:
            timestamp = parse_datetime(row["time_period_start"])
            if previous is not None and timestamp <= previous:
                raise ValueError("CoinAPI candles are not ordered oldest first")
            previous = timestamp
            # A quiet request can return nearby bars outside its requested range.
            if timestamp < cursor or timestamp + interval > timestamp_to:
                continue
            if parse_datetime(row["time_period_end"]) != timestamp + interval:
                raise ValueError("CoinAPI candle has an unexpected period")
            rows.append(
                {
                    "timestamp": timestamp,
                    **{
                        name: Decimal(str(row[key]))
                        for name, key in (
                            ("open", "price_open"),
                            ("high", "price_high"),
                            ("low", "price_low"),
                            ("close", "price_close"),
                            ("notional", "volume_traded"),
                        )
                    },
                }
            )
        if len(page) < PAGE_SIZE:
            break
        if previous is None or previous + interval <= cursor:
            raise ValueError("CoinAPI candle pagination did not advance")
        cursor = previous + interval
    frame = candles_to_data_frame(timestamp_from, timestamp_to, rows, reverse=False)
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
