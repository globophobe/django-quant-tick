import datetime
import logging
from datetime import timezone

import httpx
import pandas as pd
from pandas import DataFrame

from .candles import bybit_candles
from .constants import S3_URL

logger = logging.getLogger(__name__)


class BybitMixin:
    """Bybit mixin."""

    def get_candles(
        self, timestamp_from: datetime.datetime, timestamp_to: datetime.datetime
    ) -> DataFrame:
        """Get candles from Exchange API."""
        return bybit_candles(
            self.symbol.api_symbol,
            timestamp_from,
            timestamp_to,
            interval="1",
            limit=60,
            log_format=f"{self.log_format} validating",
        )


class BybitS3Mixin(BybitMixin):
    """Bybit S3 mixin."""

    def get_url(self, date: datetime.date) -> str:
        """Get URL."""
        symbol = self.symbol.api_symbol
        directory = f"{S3_URL}{symbol}/"
        response = httpx.get(directory)
        if response.status_code == 200:
            return f"{S3_URL}{symbol}/{symbol}{date.isoformat()}.csv.gz"
        else:
            logger.info(f"{symbol}: No data")

    def parse_dtypes_and_strip_columns(self, data_frame: DataFrame) -> DataFrame:
        """Parse dtypes and strip unnecessary columns."""
        data_frame["timestamp"] = pd.to_datetime(
            data_frame["timestamp"].astype("float"), unit="s"
        )
        data_frame["nanoseconds"] = data_frame.apply(
            lambda x: x.timestamp.nanosecond, axis=1
        )
        data_frame["timestamp"] = data_frame.apply(
            lambda x: x.timestamp.replace(nanosecond=0, tzinfo=timezone.utc), axis=1
        )
        # Before 2021-12-06, Bybit is reversed.
        first_row = data_frame.iloc[0]
        last_row = data_frame.iloc[-1]
        if first_row.timestamp > last_row.timestamp:
            data_frame = data_frame.iloc[::-1]
            data_frame.reset_index(inplace=True)
        data_frame = data_frame.rename(columns={"trdMatchID": "uid", "size": "volume"})
        return super().parse_dtypes_and_strip_columns(data_frame)
