import datetime
import logging
from typing import Optional

import pandas as pd
from pandas import DataFrame

from quant_candles.lib import (
    aggregate_trades,
    calculate_notional,
    calculate_tick_rule,
    filter_by_timestamp,
    get_current_time,
    gzip_downloader,
    set_dtypes,
    validate_data_frame,
    volume_filter_with_time_window,
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
        for (timestamp_from, timestamp_to, existing,) in iterator.iter_partition(
            self.timestamp_from,
            self.timestamp_to,
            retry=self.retry,
        ):
            date = timestamp_from.date()
            data_frame = self.get_data_frame(date)
            if data_frame is not None:
                for ts_from, ts_to in iterator.iter_hours(
                    timestamp_from,
                    timestamp_to,
                    existing,
                ):
                    df = filter_by_timestamp(data_frame, ts_from, ts_to)
                    candles = self.get_candles(ts_from, ts_to)
                    # Are there any trades?
                    if len(df):
                        if self.symbol.should_aggregate_trades:
                            df = aggregate_trades(df)
                            if self.symbol.significant_trade_filter:
                                df = volume_filter_with_time_window(
                                    df, min_volume=self.symbol.significant_trade_filter
                                )
                        else:
                            df = data_frame.drop(columns=["index"])
                    else:
                        df = pd.DataFrame([])
                    validated = validate_data_frame(
                        ts_from, ts_to, df, candles, self.symbol.should_aggregate_trades
                    )
                    self.on_data_frame(
                        self.symbol, ts_from, ts_to, df, validated=validated
                    )
            # Complete
            else:
                break

    def get_data_frame(self, date: datetime.date) -> Optional[DataFrame]:
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
