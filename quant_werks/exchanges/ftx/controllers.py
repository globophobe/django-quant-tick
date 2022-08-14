from datetime import datetime
from typing import Callable

from quant_werks.controllers import ExchangeREST
from quant_werks.models import Symbol

from .base import FTXMixin


def ftx_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: bool = False,
    verbose: bool = False,
):
    """Get FTX trades."""
    FTX(
        symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        retry=retry,
        verbose=verbose,
    ).main()


class FTX(FTXMixin, ExchangeREST):
    """FTX."""
