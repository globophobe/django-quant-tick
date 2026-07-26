from datetime import UTC, datetime
from decimal import Decimal

from pandas import DataFrame

from quant_tick.lib import (
    candles_to_data_frame,
    parse_fixed_resolution_minutes,
    resample_candles,
)

from .api import get_deribit_result, to_millis
from .constants import CANDLE_MAX_RESULTS, CANDLE_RESOLUTIONS_BY_MINUTES


def get_deribit_fetch_resolution(
    resolution: str | int | None,
) -> tuple[int, int, str]:
    """Map requested resolution to a supported Deribit interval."""
    target_minutes = parse_fixed_resolution_minutes(resolution)
    candidates = [
        minutes
        for minutes in CANDLE_RESOLUTIONS_BY_MINUTES
        if target_minutes % minutes == 0
    ]
    if not candidates:
        raise ValueError(f"Unsupported Deribit candle resolution: {resolution}")
    source_minutes = max(candidates)
    return (
        target_minutes,
        source_minutes,
        CANDLE_RESOLUTIONS_BY_MINUTES[source_minutes],
    )


def get_deribit_candle_response(
    api_symbol: str,
    start_ms: int,
    end_ms: int,
    resolution: str,
) -> dict:
    return get_deribit_result(
        "get_tradingview_chart_data",
        {
            "instrument_name": str(api_symbol).strip(),
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
            "resolution": resolution,
        },
    )


def _empty_candles() -> DataFrame:
    return DataFrame(
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "notional",
        ]
    ).set_index("timestamp")


def fetch_deribit_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    *,
    source_minutes: int,
    resolution: str,
) -> DataFrame:
    """Fetch Deribit candles."""
    start_ms = to_millis(timestamp_from)
    end_ms = to_millis(timestamp_to)
    if end_ms <= start_ms:
        return _empty_candles()

    interval_ms = source_minutes * 60_000
    max_span_ms = interval_ms * (CANDLE_MAX_RESULTS - 1)
    cursor = start_ms
    inclusive_end_ms = end_ms - 1
    rows = []
    while cursor <= inclusive_end_ms:
        chunk_end = min(cursor + max_span_ms, inclusive_end_ms)
        result = get_deribit_candle_response(
            api_symbol,
            cursor,
            chunk_end,
            resolution,
        )
        cursor = chunk_end + 1
        if result.get("status") == "no_data":
            continue
        if result.get("status") != "ok":
            raise ValueError(f"Unexpected Deribit candle status: {result.get('status')}")

        columns = {
            "timestamp": result["ticks"],
            "open": result["open"],
            "high": result["high"],
            "low": result["low"],
            "close": result["close"],
            "volume": result["cost"],
            "notional": result["volume"],
        }
        lengths = {name: len(values) for name, values in columns.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Deribit candle arrays differ in length: {lengths}")
        rows.extend(
            {
                "timestamp": datetime.fromtimestamp(timestamp / 1000, tz=UTC),
                "open": Decimal(str(open_price)),
                "high": Decimal(str(high_price)),
                "low": Decimal(str(low_price)),
                "close": Decimal(str(close_price)),
                "volume": Decimal(str(quote_cost)),
                "notional": Decimal(str(base_volume)),
            }
            for (
                timestamp,
                open_price,
                high_price,
                low_price,
                close_price,
                quote_cost,
                base_volume,
            ) in zip(
                columns["timestamp"],
                columns["open"],
                columns["high"],
                columns["low"],
                columns["close"],
                columns["volume"],
                columns["notional"],
                strict=True,
            )
        )

    if not rows:
        return _empty_candles()
    return candles_to_data_frame(
        timestamp_from,
        timestamp_to,
        rows,
        reverse=False,
    )


def deribit_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    resolution: str | int | None = "1m",
) -> DataFrame:
    """Fetch Deribit candles."""
    target_minutes, source_minutes, api_resolution = get_deribit_fetch_resolution(
        resolution
    )
    data_frame = fetch_deribit_candles(
        api_symbol,
        timestamp_from,
        timestamp_to,
        source_minutes=source_minutes,
        resolution=api_resolution,
    )
    if source_minutes == target_minutes:
        return data_frame
    return resample_candles(
        data_frame,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        resolution_minutes=target_minutes,
        source_resolution_minutes=source_minutes,
    )
