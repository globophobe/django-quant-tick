from collections.abc import Callable
from datetime import UTC, datetime, time

from quant_tick.constants import TradeDataRetry
from quant_tick.controllers import ExchangeREST
from quant_tick.models import Symbol

from .api import get_deribit_instrument_creation_timestamp
from .base import DeribitMixin


def deribit_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: TradeDataRetry = False,
    verbose: bool = False,
) -> None:
    """Get Deribit trades."""
    timestamp_from = max(timestamp_from, _get_deribit_history_start(symbol))
    if timestamp_from >= timestamp_to:
        return
    DeribitTrades(
        symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        retry=retry,
        verbose=verbose,
    ).main()


def _get_deribit_history_start(symbol: Symbol) -> datetime:
    if symbol.date_from is not None:
        return datetime.combine(symbol.date_from, time.min, tzinfo=UTC)
    creation_timestamp = get_deribit_instrument_creation_timestamp(symbol.api_symbol)
    symbol.date_from = creation_timestamp.date()
    symbol.save(update_fields=["date_from"])
    return creation_timestamp.replace(minute=0, second=0, microsecond=0)


class DeribitTrades(DeribitMixin, ExchangeREST):
    """Deribit trades REST."""
