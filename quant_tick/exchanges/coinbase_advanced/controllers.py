from collections.abc import Callable
from datetime import datetime

from quant_tick.controllers import ExchangeREST
from quant_tick.models import Symbol

from .base import CoinbaseAdvancedMixin


def coinbase_advanced_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: bool = False,
    verbose: bool = False,
) -> None:
    """Get Coinbase Advanced trades."""
    CoinbaseAdvancedTrades(
        symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        retry=retry,
        verbose=verbose,
    ).main()


class CoinbaseAdvancedTrades(CoinbaseAdvancedMixin, ExchangeREST):
    """Coinbase Advanced trades."""
