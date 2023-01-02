from datetime import datetime
from typing import Callable

from quant_candles.controllers import ExchangeREST
from quant_candles.models import Symbol

from .base import BinanceMixin


def binance_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    verbose: bool = False,
) -> None:
    """Get Binance trades."""
    BinanceTrades(
        symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        verbose=verbose,
    ).main()


class BinanceTrades(BinanceMixin, ExchangeREST):
    """Binance trades."""
