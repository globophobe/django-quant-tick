import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
from pandas import DataFrame

from quant_tick.controllers import TradeDataIterator
from quant_tick.lib import filter_by_timestamp

from .api import format_bitmex_api_timestamp, get_bitmex_api_timestamp
from .candles import bitmex_candles
from .constants import S3_URL
from .trades import get_trades


class BitmexMixin:
    """Bitmex mixin."""

    def get_uid(self, trade: dict) -> str:
        """Get uid."""
        return str(trade["trdMatchID"])

    def get_timestamp(self, trade: dict) -> datetime.datetime:
        """Get timestamp."""
        return get_bitmex_api_timestamp(trade)

    def get_nanoseconds(self, trade: dict) -> int:
        """Get nanoseconds."""
        return self.get_timestamp(trade).nanosecond

    def get_price(self, trade: dict) -> Decimal:
        """Get price."""
        return Decimal(trade["price"])

    def get_volume(self, trade: dict) -> Decimal:
        """Get volume."""
        return Decimal(trade["foreignNotional"])

    def get_notional(self, trade: dict) -> Decimal:
        """Get notional."""
        return self.get_volume(trade) / self.get_price(trade)

    def get_tick_rule(self, trade: dict) -> int:
        """Get tick rule."""
        return 1 if trade["side"] == "Buy" else -1

    def get_index(self, trade: dict) -> int:
        """Get index."""
        return np.nan  # No index, set per partition

    def get_candles(
        self, timestamp_from: datetime.datetime, timestamp_to: datetime.datetime
    ) -> DataFrame:
        """Get candles from Exchange API."""
        # Timestamp is candle close.
        return bitmex_candles(
            self.symbol.api_symbol, timestamp_from, timestamp_to, bin_size="1m"
        )


class BitmexRESTMixin(BitmexMixin):
    """Bitmex REST mixin."""

    def get_pagination_id(self, timestamp_to: datetime) -> str:
        """Get pagination_id."""
        return format_bitmex_api_timestamp(timestamp_to)

    def iter_api(self, timestamp_from: datetime, pagination_id: str) -> list:
        """Iterate Bitmex API."""
        return get_trades(
            self.symbol.api_symbol,
            timestamp_from,
            pagination_id,
            log_format=self.log_format,
        )

    def get_data_frame(self, trades: list) -> DataFrame:
        """Get data_frame."""
        data_frame = super().get_data_frame(trades)
        # No index from REST API, and trades are reversed
        data_frame["index"] = data_frame.index.values[::-1]
        return data_frame


class BitmexS3Mixin(BitmexMixin):
    """Bitmex S3 mixin."""

    def get_url(self, date: datetime.date) -> str:
        """Get CSV file url."""
        date_string = date.strftime("%Y%m%d")
        return f"{S3_URL}{date_string}.csv.gz"

    def main(self) -> None:
        """Main."""
        iterator = TradeDataIterator(self.symbol)
        exclude = [datetime.date(2025, 4, 11), datetime.date(2025, 4, 12)]
        for timestamp_from, timestamp_to, existing in iterator.iter_days(
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
                    self.on_data_frame(self.symbol, ts_from, ts_to, df, candles)
            # No data
            elif date in exclude:
                pass
            # Complete
            else:
                break

    def parse_dtypes_and_strip_columns(self, data_frame: DataFrame) -> DataFrame:
        """Parse dtypes and strip unnecessary columns.

        Bitmex data maybe accurate to the nanosecond.
        However, data is typically only provided to the microsecond.
        """
        df = data_frame.copy()
        split = df.timestamp.str.split(".", n=1, expand=True)
        dt = split[0]
        frac = split[1].fillna("")
        micro = frac.str.pad(6, side="right", fillchar="0").str[:6]
        has_nano = frac.str.len() == 9
        # Nanoseconds
        df["nanoseconds"] = 0
        df.loc[has_nano, "nanoseconds"] = frac[has_nano].str[-3:].astype("Int64")
        # Timestamp
        df["timestamp"] = pd.to_datetime(
            dt + "." + micro, format="%Y-%m-%dD%H:%M:%S.%f", utc=True
        )
        df = df.rename(columns={"trdMatchID": "uid", "foreignNotional": "volume"})
        return super().parse_dtypes_and_strip_columns(df)
