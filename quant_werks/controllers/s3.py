import datetime

import pandas as pd
from pandas import DataFrame

from quant_werks.lib import (
    aggregate_trades,
    calculate_notional,
    calculate_tick_rule,
    get_current_time,
    get_min_time,
    gzip_downloader,
    set_dtypes,
    strip_nanoseconds,
    validate_data_frame,
    volume_filter_with_time_window,
)
from quant_werks.models import AggregatedTradeData

from .base import BaseController


def use_s3():
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
        """Get columns to load CSV file."""
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
        now = get_current_time()
        max_timestamp_to = get_min_time(now, value="1t")
        for (
            daily_timestamp_from,
            daily_timestamp_to,
            daily_existing,
        ) in AggregatedTradeData.iter_days(
            self.symbol,
            self.timestamp_from,
            self.timestamp_to,
            reverse=True,
            retry=self.retry,
        ):
            date = daily_timestamp_from.date()
            url = self.get_url(date)
            data_frame = gzip_downloader(url, self.gzipped_csv_columns)
            if data_frame is not None:
                data_frame = self.filter_by_symbol(data_frame)
                data_frame = self.parse_dtypes_and_strip_columns(data_frame)
                for timestamp_from, timestamp_to in AggregatedTradeData.iter_hours(
                    daily_timestamp_from,
                    daily_timestamp_to,
                    max_timestamp_to,
                    daily_existing,
                    reverse=True,
                    retry=self.retry,
                ):
                    df = self.filter_by_timestamp(
                        data_frame, timestamp_from, timestamp_to
                    )
                    candles = self.get_candles(timestamp_from, timestamp_to)
                    # Are there any trades?
                    if len(df):
                        aggregated = aggregate_trades(df)
                        filtered = volume_filter_with_time_window(
                            aggregated, min_volume=self.symbol.min_volume
                        )
                    else:
                        filtered = pd.DataFrame([])
                    validated = validate_data_frame(
                        timestamp_from, timestamp_to, filtered, candles
                    )
                    self.on_data_frame(
                        self.symbol,
                        timestamp_from,
                        timestamp_to,
                        filtered,
                        validated=validated,
                    )
            # Complete
            else:
                break

    def filter_by_symbol(self, data_frame: DataFrame) -> DataFrame:
        """First, filter data_frame by symbol."""
        if "symbol" in data_frame.columns:
            return data_frame[data_frame.symbol == self.symbol.api_symbol]
        else:
            return data_frame

    def filter_by_timestamp(
        self,
        data_frame: DataFrame,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
    ) -> DataFrame:
        """Second, parse timestamp and filter data_frame."""
        return data_frame[
            (data_frame.timestamp >= timestamp_from)
            & (data_frame.timestamp <= timestamp_to)
        ]

    def parse_dtypes_and_strip_columns(self, data_frame: DataFrame) -> DataFrame:
        """Third, parse data_frame dtypes and strip unnecessary columns."""
        data_frame = set_dtypes(data_frame)
        data_frame = strip_nanoseconds(data_frame)
        data_frame = calculate_notional(data_frame)
        data_frame = calculate_tick_rule(data_frame)
        return data_frame[self.columns]
