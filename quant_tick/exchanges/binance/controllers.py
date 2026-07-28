import datetime
from collections.abc import Callable

from pandas import DataFrame

from quant_tick.constants import TradeDataRetry
from quant_tick.controllers import ExchangeREST, ExchangeS3
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
    """Get Binance trades."""
    kwargs = {
        "timestamp_from": timestamp_from,
        "timestamp_to": timestamp_to,
        "on_data_frame": on_data_frame,
        "retry": retry,
        "verbose": verbose,
    }
    BinanceTradesS3(symbol, **kwargs).main()
    BinanceTradesREST(symbol, **kwargs).main()


class BinanceTradesREST(BinanceMixin, ExchangeREST):
    """Binance trades via REST API."""


class BinanceTradesS3(BinanceS3Mixin, ExchangeS3):
    """Binance trades via S3 archive."""

    def get_data_frame(self, date: datetime.date) -> DataFrame | None:
        """Get data_frame from ZIP file."""
        df = zip_downloader(self.get_url(date), self.csv_columns)
        if df is not None and len(df):
            return self.parse_dtypes_and_strip_columns(df)
        return df
