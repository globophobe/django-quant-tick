from collections.abc import Callable
from datetime import datetime

from pandas import DataFrame

from quant_tick.constants import TradeDataRetry
from quant_tick.controllers import ExchangeWebSocket
from quant_tick.models import Symbol

from .candles import hyperliquid_candles


def hyperliquid_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: TradeDataRetry = False,
    verbose: bool = False,
) -> None:
    """Get Hyperliquid trades."""
    HyperliquidTradesWebSocket(
        symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        retry=retry,
        verbose=verbose,
    ).main()


class HyperliquidTradesWebSocket(ExchangeWebSocket):
    """Hyperliquid trades WebSocket."""

    def get_candles(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
    ) -> DataFrame:
        return hyperliquid_candles(
            self.symbol.api_symbol,
            timestamp_from,
            timestamp_to,
            resolution="1m",
        )
