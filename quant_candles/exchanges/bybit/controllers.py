import logging
from datetime import datetime
from typing import Callable

from quant_candles.controllers import ExchangeS3, use_s3
from quant_candles.models import Symbol

from .base import BybitS3Mixin

logger = logging.getLogger(__name__)


def bybit_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: bool = False,
    verbose: bool = False,
) -> None:
    """Get Bybit trades."""
    max_timestamp_to = use_s3()
    if timestamp_to > max_timestamp_to:
        logger.info(
            "Bybit no longer provides a paginated REST API for trades, "
            f"{timestamp_to} modified to {max_timestamp_to}"
        )
        timestamp_to = max_timestamp_to
    if timestamp_from < max_timestamp_to:
        BybitTradesS3(
            symbol,
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            on_data_frame=on_data_frame,
            retry=retry,
            verbose=verbose,
        ).main()


class BybitTradesS3(BybitS3Mixin, ExchangeS3):
    """Bybit trades S3."""
