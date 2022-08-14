from datetime import datetime
from typing import Callable

from quant_werks.controllers import ExchangeREST, ExchangeS3, use_s3
from quant_werks.models import Symbol

from .base import BitmexRESTMixin, BitmexS3Mixin


def bitmex_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: bool = False,
    verbose: bool = False,
):
    """Get BitMEX trades."""
    if timestamp_to > use_s3():
        BitmexTradesREST(
            symbol,
            timestamp_from=timestamp_from if timestamp_from > use_s3() else use_s3(),
            timestamp_to=timestamp_to,
            on_data_frame=on_data_frame,
            retry=retry,
            verbose=verbose,
        ).main()
    if timestamp_from < use_s3():
        BitmexTradesS3(
            symbol,
            timestamp_from=timestamp_from,
            timestamp_to=use_s3(),
            on_data_frame=on_data_frame,
            retry=retry,
            verbose=verbose,
        ).main()


class BitmexTradesREST(BitmexRESTMixin, ExchangeREST):
    """BitMEX trades REST."""


class BitmexTradesS3(BitmexS3Mixin, ExchangeS3):
    """BitMEX trades S3."""
