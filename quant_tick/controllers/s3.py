import datetime
import logging

import pandas as pd
from pandas import DataFrame

from quant_tick.lib import (
    calculate_notional,
    calculate_tick_rule,
    filter_by_timestamp,
    get_current_time,
    gzip_downloader,
    set_dtypes,
)

from .base import BaseController
from .iterators import TradeDataIterator

logger = logging.getLogger(__name__)


def use_s3() -> datetime:
    """Use S3."""
    date = get_current_time().date() - pd.Timedelta("2d")
    return datetime.datetime.combine(date, datetime.time.min).replace(
        tzinfo=datetime.timezone.utc
    )


class ExchangeS3(BaseController):
    """BitMEX and ByBit S3"""

    def get_url(self, date: datetime.date) -> str:
        """Get S3 url."""
        raise NotImplementedError

    @property
    def gzipped_csv_columns(self) -> list:
        """Get column names of CSV file."""
        return [
            "timestamp",
            "symbol",
            "side",
            "size",
            "price",
            "tickDirection",
            "trdMatchID",
            "grossValue",
            "foreignNotional",
        ]

    @property
    def columns(self) -> list:
        """Columns."""
        return [
            "uid",
            "timestamp",
            "nanoseconds",
            "price",
            "volume",
            "notional",
            "tickRule",
        ]

    def main(self) -> None:
        """Main."""
        iterator = TradeDataIterator(self.symbol)
        for timestamp_from, timestamp_to, existing in iterator.iter_days(
            self.timestamp_from,
            self.timestamp_to,
            retry=self.retry,
        ):
            date = timestamp_from.date()
            data_frame = self.get_data_frame(date)
            if data_frame is not None:
                df = filter_by_timestamp(data_frame, timestamp_from, timestamp_to)
                candles = self.get_candles(timestamp_from, timestamp_to)
                self.on_data_frame(self.symbol, timestamp_from, timestamp_to, df, candles)
            # Complete
            else:
                break

    def get_data_frame(self, date: datetime.date) -> DataFrame | None:
        """Get data_frame."""
        url = self.get_url(date)
        data_frame = gzip_downloader(url, self.gzipped_csv_columns)
        if data_frame is not None:
            df = self.filter_by_symbol(data_frame)
            if len(df):
                return self.parse_dtypes_and_strip_columns(df)
            return df

    def filter_by_symbol(self, data_frame: DataFrame) -> DataFrame:
        """Filter data_frame by symbol."""
        if "symbol" in data_frame.columns:
            return data_frame[data_frame.symbol == self.symbol.api_symbol]
        else:
            return data_frame

    def parse_dtypes_and_strip_columns(self, data_frame: DataFrame) -> DataFrame:
        """Parse data_frame dtypes and strip unnecessary columns."""
        data_frame = set_dtypes(data_frame)
        data_frame = calculate_notional(data_frame)
        data_frame = calculate_tick_rule(data_frame)
        return data_frame[self.columns]
