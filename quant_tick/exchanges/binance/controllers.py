import datetime
from collections.abc import Callable

from pandas import DataFrame

from quant_tick.constants import TradeDataRetry
from quant_tick.controllers import ExchangeREST, ExchangeS3, use_s3
from quant_tick.lib import zip_downloader
from quant_tick.models import Symbol

from .base import BinanceMixin, BinanceS3Mixin


def binance_trades(
    symbol: Symbol,
    timestamp_from: datetime.datetime,
    timestamp_to: datetime.datetime,
    on_data_frame: Callable,
    retry: TradeDataRetry = False,
    verbose: bool = False,
) -> None:
    """Fetch Binance spot raw trades."""
    cutoff = use_s3()
    if timestamp_to > cutoff:
        BinanceTradesREST(
            symbol,
            timestamp_from=max(timestamp_from, cutoff),
            timestamp_to=timestamp_to,
            on_data_frame=on_data_frame,
            retry=retry,
            verbose=verbose,
        ).main()
    if timestamp_from < cutoff:
        BinanceTradesS3(
            symbol,
            timestamp_from=timestamp_from,
            timestamp_to=min(timestamp_to, cutoff),
            on_data_frame=on_data_frame,
            retry=retry,
            verbose=verbose,
        ).main()


class BinanceTradesREST(BinanceMixin, ExchangeREST):
    """Binance spot raw trades via REST API."""


class BinanceTradesS3(BinanceS3Mixin, ExchangeS3):
    """Binance spot raw trades via Data Vision archives."""

    def get_data_frame(self, date: datetime.date) -> DataFrame | None:
        df = zip_downloader(self.get_url(date), self.csv_columns)
        if df is not None and len(df):
            return self.parse_dtypes_and_strip_columns(df)
        return df
