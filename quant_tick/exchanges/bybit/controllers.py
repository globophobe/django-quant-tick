from collections.abc import Callable
from datetime import date, datetime

from quant_tick.constants import Frequency, TradeDataRetry
from quant_tick.controllers import (
    ChunkedExchangeS3,
    ExchangeWebSocket,
)
from quant_tick.controllers.s3 import get_archive_followup_start
from quant_tick.lib import gzip_chunk_downloader
from quant_tick.models import Symbol, TradeData

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
    """Use native archives first, then recover retained WebSocket minutes."""
    category = validate_bybit_trade_symbol(symbol)
    kwargs = {
        "timestamp_from": timestamp_from,
        "timestamp_to": timestamp_to,
        "on_data_frame": on_data_frame,
        "retry": retry,
        "verbose": verbose,
    }
    followup_from = get_archive_followup_start(timestamp_from)
    followup = (
        BybitTradesWebSocket(symbol, **{**kwargs, "timestamp_from": followup_from})
        if followup_from < timestamp_to
        else None
    )
    websocket_rows = (
        followup.get_websocket_rows(followup_from, timestamp_to)
        if followup is not None and not retry
        else None
    )
    archive = BybitSpotTradesS3 if category == "spot" else BybitTradesS3
    archive(symbol, **kwargs).main()
    if followup is not None:
        followup.main(websocket_rows=websocket_rows)


class BybitTradesS3(BybitS3Mixin, ChunkedExchangeS3):
    """Bybit trades S3."""

    archive_chunksize = 200_000
    allow_descending_archive = True

    def get_data_frame_chunks(self, value: date):
        """Download one complete archive and return its CSV chunks."""
        return gzip_chunk_downloader(
            self.get_url(value),
            self.gzipped_csv_columns,
            chunksize=self.archive_chunksize,
        )


class BybitSpotTradesS3(BybitSpotS3Mixin, ChunkedExchangeS3):
    """Bybit spot trades S3."""

    allow_descending_archive = True

    def get_data_frame_chunks(self, value: date):
        return gzip_chunk_downloader(
            self.get_url(value),
            self.gzipped_csv_columns,
            chunksize=self.archive_chunksize,
        )


class BybitTradesWebSocket(BybitMixin, ExchangeWebSocket):
    """Bybit trades WebSocket."""

    def validate_websocket_partition(self, row, timestamp_to, candles):
        # Retain validated history when retained WebSocket rows are incomplete.
        if TradeData.objects.filter(
            symbol=self.symbol,
            timestamp=row.timestamp,
            frequency=Frequency.MINUTE,
            ok=True,
        ).exists():
            return None
        return super().validate_websocket_partition(row, timestamp_to, candles)
