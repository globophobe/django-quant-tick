"""HyperCore OHLCV in quote units, retaining missing quote turnover and gaps.

Femtos substitutes native Hyperliquid candles over some archive gaps; those
bars have no quote turnover. They must not become zero-volume observations.
"""

import pandas as pd
from pandas import DataFrame

from quant_tick.lib import parse_fixed_resolution_minutes

from .api import (
    PAGE_SIZE,
    completed_windows,
    get_femtos_response,
    normalize_coin,
    number,
    ordered_points,
)

TIMEFRAMES = {
    1: "1m",
    3: "3m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "1h",
    120: "2h",
    240: "4h",
    480: "8h",
    720: "12h",
    1440: "1d",
}


def femtos_candles(api_symbol, timestamp_from, timestamp_to, resolution="1m"):
    coin = normalize_coin(api_symbol)
    minutes = parse_fixed_resolution_minutes(resolution)
    if minutes not in TIMEFRAMES:
        raise ValueError("Unsupported Femtos candle resolution")
    timeframe = TIMEFRAMES[minutes]
    rows = []
    for start, end in completed_windows(timestamp_from, timestamp_to, minutes):
        result = get_femtos_response(
            "/candles",
            {
                "layer": "hypercore",
                "market": coin,
                "tf": timeframe,
                "from": int(start.timestamp() * 1000),
                "to": int(end.timestamp() * 1000),
                "limit": PAGE_SIZE,
                "fill": "false",
            },
        )
        if (result["layer"], result["market"], result["tf"]) != (
            "hypercore",
            coin,
            timeframe,
        ):
            raise ValueError("Femtos returned an unexpected candle market or interval")
        head = result.get("asOf")
        if result["candles"] and head is None:
            raise ValueError("Femtos candles have no indexed head")
        for timestamp, point in ordered_points(result["candles"], start, end, minutes):
            if timestamp + pd.Timedelta(minutes=minutes) > pd.Timestamp(
                head, unit="ms", tz="UTC"
            ):
                continue
            prices = {
                name: number(point[key])
                for name, key in (
                    ("open", "o"),
                    ("high", "h"),
                    ("low", "l"),
                    ("close", "c"),
                )
            }
            if not (
                0
                < prices["low"]
                <= min(prices["open"], prices["close"])
                <= max(prices["open"], prices["close"])
                <= prices["high"]
            ):
                raise ValueError("Femtos candle has invalid OHLC bounds")
            rows.append(
                {
                    "timestamp": timestamp,
                    **prices,
                    "notional": number(point["v"]),
                    "volume": number(point["vq"], nullable=True),
                    "ticks": number(point["n"]),
                    "candle_source": "femtos",
                }
            )
    return DataFrame(
        rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "notional",
            "volume",
            "ticks",
            "candle_source",
        ],
    ).set_index("timestamp")
