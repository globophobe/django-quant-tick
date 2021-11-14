import datetime

import pandas as pd
from pandas import DataFrame

from cryptofeed_werks.lib import (
    calculate_notional,
    calculate_tick_rule,
    get_current_time,
    gzip_downloader,
    set_dtypes,
    strip_nanoseconds,
    utc_timestamp,
)

from .base import BaseController


def use_s3():
    date = get_current_time().date() - pd.Timedelta("2d")
    return datetime.combine(date, datetime.time.min).replace(
        tzinfo=datetime.timezone.utc
    )


class ExchangeS3(BaseController):
    """BitMEX and ByBit S3"""

    def get_url(self, date: datetime.date) -> str:
        """Get S3 url."""
        raise NotImplementedError

    @property
    def get_columns(self) -> list:
        """Get columns to load CSV file."""
        raise NotImplementedError

    def main(self) -> None:
        for timestamp_from, timestamp_to, is_complete in self.iter_timeframe:
            date = timestamp_from.date()
            url = self.get_url(date)
            data_frame = gzip_downloader(url, self.get_columns)
            if data_frame is not None:
                df = self.filter_by_symbol(data_frame)
                if len(df):
                    df = self.parse_and_filter_by_timestamp(df)
                    df = self.parse_dtypes_and_strip_columns(df)
                    self.on_data_frame(df, is_complete)

    def filter_by_symbol(self, data_frame: DataFrame) -> DataFrame:
        """First, filter data_frame by symbol."""
        if "symbol" in data_frame.columns:
            return data_frame[data_frame.symbol == self.symbol.name]
        else:
            return data_frame

    def parse_and_filter_by_timestamp(
        self, data_frame: DataFrame, timestamp_from: datetime, timestamp_to: datetime
    ) -> DataFrame:
        """Second, parse timestamp and filter data_frame."""
        data_frame = utc_timestamp(data_frame)
        data_frame = strip_nanoseconds(data_frame)
        return data_frame[
            (data_frame.timestamp >= timestamp_from)
            & (data_frame.timestamp <= timestamp_to)
        ]

    def parse_dtypes_and_strip_columns(self, data_frame: DataFrame) -> DataFrame:
        """Third, parse data_frame dtypes and strip unnecessary columns."""
        data_frame = set_dtypes(data_frame)
        data_frame = calculate_notional(data_frame)
        data_frame = calculate_tick_rule(data_frame)
        return data_frame[self.columns]


class ExchangeMultiSymbolS3(BaseController):
    def filter_by_symbol(self, data_frame: DataFrame) -> DataFrame:
        """First, filter data_frame by multiple symbols."""
        if "symbol" in data_frame.columns:
            query = " | ".join(
                [f'symbol == "{s["symbol"]}"' for s in self.active_symbols]
            )
            return data_frame.query(query)
        else:
            return data_frame
