from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.lib import (
    candles_to_data_frame,
    iter_chunks,
    parse_fixed_resolution_minutes,
    resample_candles,
)

from .api import get_phoenix_response, to_millis
from .constants import CANDLE_MAX_RESULTS, CANDLE_RESOLUTIONS_BY_MINUTES


def phoenix_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    resolution: str | int | None = "1m",
) -> DataFrame:
    """Fetch completed exchange candles, retaining quote/base and mark-price units.

    External candle sources are disabled. Phoenix fills empty buckets with the
    prior close, including ranges before its first trade; set Symbol.date_from
    to the market's observed trading start when backfilling.
    """
    target = parse_fixed_resolution_minutes(resolution)
    source = max(
        minutes for minutes in CANDLE_RESOLUTIONS_BY_MINUTES if target % minutes == 0
    )
    interval = pd.Timedelta(minutes=source)
    fields = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volumeQuote",
        "notional": "volume",
        "mark_open": "markOpen",
        "mark_high": "markHigh",
        "mark_low": "markLow",
        "mark_close": "markClose",
    }
    rows = []
    for start, end in iter_chunks(
        timestamp_from, timestamp_to, value=interval * (CANDLE_MAX_RESULTS - 1)
    ):
        data = get_phoenix_response(
            f"/v1/candles/{api_symbol.strip().upper()}",
            {
                "timeframe": CANDLE_RESOLUTIONS_BY_MINUTES[source],
                "startTime": to_millis(start),
                "endTime": to_millis(end) - 1,
                "enableExternalSource": "false",
            },
        )
        for candle in data:
            timestamp = datetime.fromtimestamp(candle["time"] / 1000, tz=UTC)
            if not start <= timestamp < end or timestamp + interval > timestamp_to:
                continue
            if candle.get("externalSource"):
                raise ValueError("Phoenix returned an external-source candle")
            rows.append(
                {
                    "timestamp": timestamp,
                    **{
                        name: Decimal(str(candle[key]))
                        if candle.get(key) is not None
                        else None
                        for name, key in fields.items()
                    },
                    "trades": candle.get("tradeCount"),
                }
            )
    if not rows:
        return DataFrame(columns=["timestamp", *fields, "trades"]).set_index(
            "timestamp"
        )
    frame = candles_to_data_frame(timestamp_from, timestamp_to, rows, reverse=False)
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Phoenix candle timestamps are not strictly increasing")
    if source == target:
        return frame
    kwargs = {
        "timestamp_from": timestamp_from,
        "timestamp_to": timestamp_to,
        "resolution_minutes": target,
        "source_resolution_minutes": source,
    }
    result = resample_candles(frame, **kwargs)
    mark_columns = {f"mark_{name}": name for name in ("open", "high", "low", "close")}
    marks = resample_candles(
        frame[list(mark_columns)].rename(columns=mark_columns), **kwargs
    )
    result[list(mark_columns)] = marks.add_prefix("mark_")
    result["trades"] = (
        frame["trades"]
        .groupby(frame.index.floor(f"{target}min"))
        .sum(min_count=target // source)
        .reindex(result.index)
    )
    return result
