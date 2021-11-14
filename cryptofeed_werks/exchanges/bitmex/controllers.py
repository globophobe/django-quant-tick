from datetime import datetime
from typing import Callable

import pandas as pd

from cryptofeed_werks.controllers import (
    ExchangeMultiSymbolREST,
    ExchangeMultiSymbolS3,
    ExchangeREST,
    ExchangeS3,
    use_s3,
)
from cryptofeed_werks.models import Symbol

from .base import BitmexFuturesS3Mixin, BitmexMixin, BitmexS3Mixin


def bitmex_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    verbose: bool = False,
):
    if timestamp_to > use_s3():
        BitmexTradesREST(
            symbol,
            timestamp_from=timestamp_from if timestamp_from < use_s3() else use_s3(),
            timestamp_to=timestamp_to,
            on_data_frame=on_data_frame,
            verbose=verbose,
        ).main()
    if timestamp_from < use_s3():
        BitmexTradesS3(
            symbol,
            timestamp_from=timestamp_from,
            timestamp_to=use_s3() - pd.Timedelta("1d"),
            on_data_frame=on_data_frame,
            verbose=verbose,
        ).main()


def bitmex_futures(
    symbol: str = None,
    timestamp_from: str = None,
    timestamp_to: str = None,
    verbose: bool = False,
):
    if timestamp_to() > use_s3():
        BitmexFuturesREST(
            symbol,
            timestamp_from=timestamp_from if timestamp_from < use_s3() else use_s3(),
            timestamp_to=timestamp_to,
            verbose=verbose,
        )
    BitmexFuturesS3(
        symbol,
        timestamp_from=timestamp_from,
        timestamp_to=use_s3() - pd.Timedelta("1d"),
        verbose=verbose,
    ).main()


class BitmexTradesREST(BitmexMixin, ExchangeREST):
    pass


class BitmexTradesS3(BitmexS3Mixin, ExchangeS3):
    pass


class BitmexFuturesREST(BitmexMixin, ExchangeMultiSymbolREST):
    pass


class BitmexFuturesS3(BitmexFuturesS3Mixin, ExchangeMultiSymbolS3):
    def has_data(self, date):
        # No active symbols 2016-10-01 to 2016-10-25.
        return super().has_data(date)
