from datetime import datetime
from typing import Callable

import pandas as pd

from cryptofeed_werks.controllers import ExchangeREST, ExchangeS3, use_s3
from cryptofeed_werks.models import Symbol

from .base import BybitRESTMixin, BybitS3Mixin


def bybit_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    verbose: bool = False,
):
    if timestamp_to > use_s3():
        BybitTrades(
            symbol,
            timestamp_from=timestamp_from if timestamp_from < use_s3() else use_s3(),
            timestamp_to=timestamp_to,
            on_data_frame=on_data_frame,
            verbose=verbose,
        ).main()
    if timestamp_from < use_s3():
        BybitTradesS3(
            symbol,
            timestamp_from=timestamp_from,
            timestamp_to=use_s3() - pd.Timedelta("1d"),
            on_data_frame=on_data_frame,
            verbose=verbose,
        ).main()


class BybitTrades(BybitRESTMixin, ExchangeREST):
    pass


class BybitTradesS3(BybitS3Mixin, ExchangeS3):
    pass
