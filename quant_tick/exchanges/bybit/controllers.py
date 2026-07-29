from collections.abc import Callable
from datetime import date, datetime

from quant_tick.constants import TradeDataRetry
from quant_tick.controllers import (
    ChunkedExchangeS3,
    ExchangeWebSocket,
)
from quant_tick.lib import gzip_chunk_downloader
from quant_tick.models import Symbol

from .base import (
    BybitMixin,
    BybitS3Mixin,
    BybitSpotS3Mixin,
    validate_bybit_trade_symbol,
)


def bybit_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: TradeDataRetry = False,
    verbose: bool = False,
) -> None:
    """Get Bybit trades."""
    category = validate_bybit_trade_symbol(symbol)
    archive = BybitSpotTradesS3 if category == "spot" else BybitTradesS3
    archive(
        symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        retry=retry,
        verbose=verbose,
    ).main()
    BybitTradesWebSocket(
        symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        retry=retry,
        verbose=verbose,
    ).main()


class BybitTradesS3(BybitS3Mixin, ChunkedExchangeS3):
    """Bybit trades S3."""

    archive_chunksize = 200_000

    def get_data_frame_chunks(self, value: date):
        """Download one complete archive and return its CSV chunks."""
        return gzip_chunk_downloader(
            self.get_url(value),
            self.gzipped_csv_columns,
            chunksize=self.archive_chunksize,
        )



class BybitSpotTradesS3(BybitSpotS3Mixin, ChunkedExchangeS3):
    """Bybit spot trades S3."""

    def get_data_frame_chunks(self, value: date):
        return gzip_chunk_downloader(
            self.get_url(value),
            self.gzipped_csv_columns,
            chunksize=self.archive_chunksize,
        )


class BybitTradesWebSocket(BybitMixin, ExchangeWebSocket):
    """Bybit trades WebSocket."""
