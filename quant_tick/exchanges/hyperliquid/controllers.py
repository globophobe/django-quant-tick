from collections.abc import Callable
from datetime import datetime
from functools import cached_property

from pandas import DataFrame

from quant_tick.constants import Frequency, SymbolType, TradeDataRetry
from quant_tick.controllers import ExchangeWebSocket
from quant_tick.models import Symbol, TradeData

from ..coinapi.repair import CoinAPIRepairTrades
from .candles import (
    fill_coinapi_candles,
    hyperliquid_candles,
    native_hyperliquid_candles,
)
from .sources import coinapi_symbol


def hyperliquid_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: TradeDataRetry = False,
    verbose: bool = False,
) -> None:
    """Keep native trades where usable; backfill gaps with optional CoinAPI."""
    controller = HyperliquidTradesWebSocket
    if symbol.symbol_type == SymbolType.PERPETUAL and coinapi_symbol(symbol.api_symbol):
        controller = HyperliquidTrades
    controller(
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
            allow_coinapi=False,
        )

    def validate_websocket_partition(self, row, timestamp_to, candles):
        # Removing a vendor key must not downgrade previously validated history.
        if TradeData.objects.filter(
            symbol=self.symbol,
            timestamp=row.timestamp,
            frequency=Frequency.MINUTE,
            ok=True,
        ).exists():
            return None
        frames = super().validate_websocket_partition(row, timestamp_to, candles)
        if frames is not None:
            for frame in frames:
                if frame is not None:
                    frame["data_source"] = "hyperliquid"
        return frames


class HyperliquidTrades(CoinAPIRepairTrades, HyperliquidTradesWebSocket):
    """Keep native minutes and use CoinAPI for missing trades and older candles."""

    @cached_property
    def api_symbol(self):
        return coinapi_symbol(self.symbol.api_symbol)

    def get_native_candles(self, timestamp_from, timestamp_to):
        return native_hyperliquid_candles(
            self.symbol.api_symbol, timestamp_from, timestamp_to
        )

    def get_validation_candles(self, timestamp_from, timestamp_to, native):
        return fill_coinapi_candles(
            self.symbol.api_symbol,
            timestamp_from,
            timestamp_to,
            native,
            resolution="1m",
        )
