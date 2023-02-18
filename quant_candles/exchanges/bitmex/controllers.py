import datetime
from typing import Callable, Optional

import pandas as pd
from pandas import DataFrame

from quant_candles.controllers import ExchangeREST, ExchangeS3, use_s3
from quant_candles.models import Symbol

from .api import get_bitmex_api_response
from .base import BitmexRESTMixin, BitmexS3Mixin
from .constants import API_URL, XBTUSD


def bitmex_trades(
    symbol: Symbol,
    timestamp_from: datetime.datetime,
    timestamp_to: datetime.datetime,
    on_data_frame: Callable,
    retry: bool = False,
    verbose: bool = False,
) -> None:
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
            timestamp_to=timestamp_to if timestamp_to < use_s3() else use_s3(),
            on_data_frame=on_data_frame,
            retry=retry,
            verbose=verbose,
        ).main()


class BitmexTradesREST(BitmexRESTMixin, ExchangeREST):
    """BitMEX trades REST."""


class BitmexTradesS3(BitmexS3Mixin, ExchangeS3):
    """BitMEX trades S3."""

    def get_data_frame(self, date: datetime.date) -> Optional[DataFrame]:
        """Get data_frame.

        Downloaded file has multiple symbols. Do nothing before listing date.
        """
        base_url = f"{API_URL}/instrument?symbol={self.symbol.api_symbol}&count=1"

        def get_api_url(*args, **kwargs):
            return base_url

        # In the case of XBTUSD, the listing date was previously correct, but has been
        # revised by BitMEX to 2016-05-13.
        if self.symbol.api_symbol == XBTUSD:
            listing_date = datetime.date(2015, 9, 25)
        else:
            data = get_bitmex_api_response(get_api_url, base_url)
            listing_date = pd.to_datetime(data[0]["listing"]).date()

        # Without this check, empty data frames may be acquired from BitMEX data before
        # the symbol listing date.
        if date >= listing_date:
            return super().get_data_frame(date)
