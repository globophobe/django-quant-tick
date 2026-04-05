from collections.abc import Callable
from datetime import datetime

from django.db import models
from pandas import DataFrame


class BaseController:
    """Base controller for exchange trade and candle ingestion."""

    def __init__(
        self,
        symbol: models.Model,
        timestamp_from: datetime,
        timestamp_to: datetime,
        on_data_frame: Callable,
        retry: bool = False,
        verbose: bool = True,
    ) -> None:
        """Store the controller inputs for one symbol and time range."""
        self.symbol = symbol
        self.timestamp_from = timestamp_from
        self.timestamp_to = timestamp_to
        self.on_data_frame = on_data_frame
        self.retry = retry
        self.verbose = verbose

    @property
    def log_format(self) -> str:
        """Log format string for progress messages."""
        symbol = str(self.symbol)
        return f"{symbol}: {{timestamp}}"

    @property
    def columns(self) -> list:
        """Canonical trade-data column order."""
        return [
            "uid",
            "timestamp",
            "nanoseconds",
            "price",
            "volume",
            "notional",
            "tickRule",
            "index",
        ]

    def main(self) -> None:
        """Fetch data and pass normalized frames to on_data_frame."""
        raise NotImplementedError

    def get_candles(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> DataFrame:
        """Fetch exchange candle data for validation."""
        raise NotImplementedError
