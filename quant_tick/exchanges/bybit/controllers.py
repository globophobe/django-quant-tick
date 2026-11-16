from collections.abc import Callable
from datetime import date, datetime
from functools import cached_property

from quant_tick.constants import Frequency, TradeDataRetry
from quant_tick.controllers import (
    ChunkedExchangeS3,
    ExchangeWebSocket,
)
from quant_tick.controllers.s3 import get_archive_followup_start
from quant_tick.lib import gzip_chunk_downloader
from quant_tick.models import Symbol, TradeData

from ..coinapi.repair import CoinAPIRepairTrades
from .base import (
    BybitMixin,
    BybitS3Mixin,
    BybitSpotS3Mixin,
    validate_bybit_trade_symbol,
)
from .sources import coinapi_symbol


def bybit_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: TradeDataRetry = False,
    verbose: bool = False,
) -> None:
    """Use native archives first, then WebSocket minutes and optional paid repair."""
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
    controller = BybitTradesWebSocket
    followup_from = get_archive_followup_start(timestamp_from)
    if coinapi_symbol(symbol):
        controller = BybitTrades
        followup_from = timestamp_from
    if followup_from < timestamp_to:
        controller(
            symbol,
            timestamp_from=followup_from,
            timestamp_to=timestamp_to,
            on_data_frame=on_data_frame,
            retry=retry,
            verbose=verbose,
        ).main()


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
        # Retain validated repairs even when the optional key is later removed.
        if TradeData.objects.filter(
            symbol=self.symbol,
            timestamp=row.timestamp,
            frequency=Frequency.MINUTE,
            ok=True,
        ).exists():
            return None
        frames = super().validate_websocket_partition(row, timestamp_to, candles)
        if frames is not None:
            for frame in frames:
                if frame is not None:
                    frame["data_source"] = "bybit"
        return frames


class BybitTrades(CoinAPIRepairTrades, BybitTradesWebSocket):
    """Repair uncovered minutes without changing the native market identity.

    Bybit candles validate both sources; complete archives and WebSocket minutes
    need no paid requests. Only spot and USDT perpetual mappings are supported.
    """

    @cached_property
    def api_symbol(self):
        return coinapi_symbol(self.symbol)

    def get_native_candles(self, timestamp_from, timestamp_to):
        return BybitMixin.get_candles(self, timestamp_from, timestamp_to)
