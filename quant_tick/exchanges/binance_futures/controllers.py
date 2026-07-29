import datetime
from collections.abc import Callable
from functools import wraps

from pandas import DataFrame

from quant_tick.constants import TradeDataRetry
from quant_tick.controllers import ChunkedExchangeS3, ExchangeREST
from quant_tick.lib import zip_chunk_downloader
from quant_tick.models import Symbol

from .base import BinanceFuturesMixin, BinanceFuturesS3Mixin


def _store_as_aggregated(on_data_frame: Callable) -> Callable:
    @wraps(on_data_frame)
    def callback(
        symbol: Symbol,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
        data_frame: DataFrame,
        candles: DataFrame,
        **kwargs,
    ):
        if not kwargs:
            kwargs["aggregated_trades"] = data_frame
        return on_data_frame(
            symbol,
            timestamp_from,
            timestamp_to,
            data_frame,
            candles,
            **kwargs,
        )

    return callback


def binance_futures_trades(
    symbol: Symbol,
    timestamp_from: datetime.datetime,
    timestamp_to: datetime.datetime,
    on_data_frame: Callable,
    retry: TradeDataRetry = False,
    verbose: bool = False,
) -> None:
    """Get Binance Futures trades."""
    kwargs = {
        "timestamp_from": timestamp_from,
        "timestamp_to": timestamp_to,
        "on_data_frame": _store_as_aggregated(on_data_frame),
        "retry": retry,
        "verbose": verbose,
    }
    BinanceFuturesTradesS3(symbol, **kwargs).main()
    BinanceFuturesTradesREST(symbol, **kwargs).main()


class BinanceFuturesTradesREST(BinanceFuturesMixin, ExchangeREST):
    """Binance Futures trades REST."""


class BinanceFuturesTradesS3(BinanceFuturesS3Mixin, ChunkedExchangeS3):
    """Binance Futures trades S3."""

    def get_data_frame_chunks(self, value: datetime.date):
        return zip_chunk_downloader(
            self.get_url(value),
            self.csv_columns,
            chunksize=self.archive_chunksize,
        )
