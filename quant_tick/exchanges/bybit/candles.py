from datetime import UTC, datetime
from decimal import Decimal

from pandas import DataFrame

from quant_tick.constants import SymbolType
from quant_tick.lib import (
    candles_to_data_frame,
    get_interval_inclusive_end,
    parse_fixed_resolution_minutes,
    resample_candles,
)

from .api import get_bybit_result, to_millis
from .constants import CANDLE_MAX_RESULTS, CANDLE_RESOLUTIONS_BY_MINUTES


def get_bybit_category(api_symbol: str, symbol_type: str) -> str:
    if symbol_type == SymbolType.SPOT:
        return "spot"
    if symbol_type != SymbolType.PERPETUAL:
        raise ValueError(f"Unsupported Bybit symbol type: {symbol_type}")
    symbol = str(api_symbol).upper()
    return "linear" if symbol.endswith(("USDT", "USDC")) else "inverse"


def get_bybit_fetch_resolution(
    resolution: str | int | None,
) -> tuple[int, int, str]:
    """Map requested resolution to a supported Bybit interval."""
    target_minutes = parse_fixed_resolution_minutes(resolution)
    candidates = [
        minutes
        for minutes in CANDLE_RESOLUTIONS_BY_MINUTES
        if target_minutes % minutes == 0
    ]
    if not candidates:
        raise ValueError(f"Unsupported Bybit candle resolution: {resolution}")
    source_minutes = max(candidates)
    return (
        target_minutes,
        source_minutes,
        CANDLE_RESOLUTIONS_BY_MINUTES[source_minutes],
    )


def get_bybit_candle_response(
    api_symbol: str,
    start_ms: int,
    end_ms: int,
    interval: str,
    category: str,
    limit: int = CANDLE_MAX_RESULTS,
) -> dict:
    return get_bybit_result(
        "/v5/market/kline",
        {
            "category": category,
            "symbol": str(api_symbol).strip(),
            "interval": interval,
            "start": start_ms,
            "end": end_ms,
            "limit": limit,
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


def fetch_bybit_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    *,
    source_minutes: int,
    interval: str,
    category: str,
) -> DataFrame:
    """Fetch Bybit candles."""
    if timestamp_to <= timestamp_from:
        return _empty_candles()

    inclusive_end = get_interval_inclusive_end(
        timestamp_from,
        timestamp_to,
        source_minutes,
    )
    start_ms = to_millis(timestamp_from)
    cursor_end_ms = to_millis(inclusive_end)
    rows = []
    while cursor_end_ms >= start_ms:
        result = get_bybit_candle_response(
            api_symbol,
            start_ms,
            cursor_end_ms,
            interval,
            category,
        )
        page = result.get("list", [])
        if not page:
            break

        for candle in page:
            if category == "inverse":
                quote_volume = candle[5]
                base_volume = candle[6]
            else:
                base_volume = candle[5]
                quote_volume = candle[6]
            rows.append(
                {
                    "timestamp": datetime.fromtimestamp(
                        int(candle[0]) / 1000,
                        tz=UTC,
                    ),
                    "open": Decimal(str(candle[1])),
                    "high": Decimal(str(candle[2])),
                    "low": Decimal(str(candle[3])),
                    "close": Decimal(str(candle[4])),
                    "volume": Decimal(str(quote_volume)),
                    "notional": Decimal(str(base_volume)),
                }
            )

        oldest_ms = int(page[-1][0])
        if oldest_ms <= start_ms:
            break
        next_end_ms = oldest_ms - 1
        if next_end_ms >= cursor_end_ms:
            raise ValueError("Bybit candle pagination did not move backward")
        cursor_end_ms = next_end_ms

    if not rows:
        return _empty_candles()
    return candles_to_data_frame(timestamp_from, timestamp_to, rows)


def bybit_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    resolution: str | int | None = "1m",
    symbol_type: str = SymbolType.PERPETUAL,
) -> DataFrame:
    """Fetch Bybit candles."""
    target_minutes, source_minutes, interval = get_bybit_fetch_resolution(resolution)
    data_frame = fetch_bybit_candles(
        api_symbol,
        timestamp_from,
        timestamp_to,
        source_minutes=source_minutes,
        interval=interval,
        category=get_bybit_category(api_symbol, symbol_type),
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
