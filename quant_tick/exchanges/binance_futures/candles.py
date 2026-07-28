from datetime import datetime

from pandas import DataFrame

from quant_tick.constants import SymbolType
from quant_tick.exchanges.binance.candles import binance_candles


def binance_futures_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    *,
    resolution: str | int | None = None,
    interval: str = "1m",
    limit: int | None = None,
    log_format: str | None = None,
) -> DataFrame:
    """Fetch Binance Futures candles."""
    return binance_candles(
        api_symbol,
        timestamp_from,
        timestamp_to,
        resolution=resolution,
        interval=interval,
        symbol_type=SymbolType.PERPETUAL,
        limit=limit,
        log_format=log_format,
    )
