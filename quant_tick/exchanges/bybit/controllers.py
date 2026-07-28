from collections.abc import Callable
from datetime import datetime

from quant_tick.constants import TradeDataRetry
from quant_tick.controllers import ExchangeS3, ExchangeWebSocket
from quant_tick.models import Symbol

from .base import (
    BybitMixin,
    BybitS3Mixin,
    BybitSpotS3Mixin,
    validate_bybit_trade_symbol,
)


def bybit_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: TradeDataRetry = False,
    verbose: bool = False,
) -> None:
    """Get Bybit trades."""
    category = validate_bybit_trade_symbol(symbol)
    archive = BybitSpotTradesS3 if category == "spot" else BybitTradesS3
    archive(
        symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        retry=retry,
        verbose=verbose,
    ).main()
    BybitTradesWebSocket(
        symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        retry=retry,
        verbose=verbose,
    ).main()


class BybitTradesS3(BybitS3Mixin, ExchangeS3):
    """Bybit trades S3."""


class BybitSpotTradesS3(BybitSpotS3Mixin, ExchangeS3):
    """Bybit spot trades S3."""


class BybitTradesWebSocket(BybitMixin, ExchangeWebSocket):
    """Bybit trades WebSocket."""
