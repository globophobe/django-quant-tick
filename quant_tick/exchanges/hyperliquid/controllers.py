from collections.abc import Callable
from datetime import datetime

from pandas import DataFrame

from quant_tick.constants import Frequency, TradeDataRetry
from quant_tick.controllers import ExchangeWebSocket
from quant_tick.controllers.repair import ExchangeRepair
from quant_tick.models import Symbol, TradeData

from ..quicknode.api import available, history_start
from ..quicknode.candles import validation_candles
from ..quicknode.trades import QuickNodeTrades
from .candles import (
    fill_quicknode_candles,
    hyperliquid_candles,
    native_hyperliquid_candles,
)


def hyperliquid_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: TradeDataRetry = False,
    verbose: bool = False,
) -> None:
    """Keep native spot/perpetual trades; backfill gaps with optional QuickNode."""
    controller = HyperliquidTradesWebSocket
    if available():
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
            allow_quicknode=False,
        )

    def validate_websocket_partition(
        self, row, timestamp_to, candles, *, validated_only=True
    ):
        # Removing a vendor key must not downgrade previously validated history.
        if TradeData.objects.filter(
            symbol=self.symbol,
            timestamp=row.timestamp,
            frequency=Frequency.MINUTE,
            ok=True,
        ).exists():
            return None
        return super().validate_websocket_partition(
            row, timestamp_to, candles, validated_only=validated_only
        )


class HyperliquidTrades(QuickNodeTrades, ExchangeRepair, HyperliquidTradesWebSocket):
    """Keep native minutes and use QuickNode for missing trades and older candles."""

    def history_start(self):
        return history_start()

    def get_native_candles(self, timestamp_from, timestamp_to):
        return native_hyperliquid_candles(
            self.symbol.api_symbol, timestamp_from, timestamp_to
        )

    def get_validation_candles(self, timestamp_from, timestamp_to, native):
        return fill_quicknode_candles(
            self.symbol.api_symbol,
            timestamp_from,
            timestamp_to,
            native,
            resolution="1m",
            fetch=validation_candles,
        )
