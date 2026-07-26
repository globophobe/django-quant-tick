import datetime
from collections.abc import Callable
from functools import wraps

from pandas import DataFrame

from quant_tick.constants import TradeDataRetry
from quant_tick.controllers import ExchangeREST, ExchangeS3
from quant_tick.lib import zip_downloader
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
    """Fetch public Binance USD-M aggregate trades."""
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
    """Binance USD-M aggregate trades via public REST API."""


class BinanceFuturesTradesS3(BinanceFuturesS3Mixin, ExchangeS3):
    """Binance USD-M aggregate trades via Data Vision archives."""

    def get_data_frame(self, date: datetime.date) -> DataFrame | None:
        df = zip_downloader(self.get_url(date), self.csv_columns)
        if df is not None and len(df):
            return self.parse_dtypes_and_strip_columns(df)
        return df
