from datetime import datetime
from typing import Callable

from cryptofeed_werks.controllers import ExchangeREST
from cryptofeed_werks.models import Symbol

from .base import BitfinexMixin


def bitfinex_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    verbose: bool = False,
):
    BitfinexTrades(
        symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        verbose=verbose,
    ).main()


class BitfinexTrades(BitfinexMixin, ExchangeREST):
    pass
