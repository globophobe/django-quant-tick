import datetime
from collections.abc import Callable

from quant_tick.constants import TradeDataRetry
from quant_tick.controllers import ChunkedExchangeS3, ExchangeREST
from quant_tick.lib import zip_chunk_downloader
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


class BinanceTradesS3(BinanceS3Mixin, ChunkedExchangeS3):
    """Binance trades via S3 archive."""

    def get_data_frame_chunks(self, value: datetime.date):
        return zip_chunk_downloader(
            self.get_url(value),
            self.csv_columns,
            chunksize=self.archive_chunksize,
            usecols=self.archive_csv_columns,
        )
