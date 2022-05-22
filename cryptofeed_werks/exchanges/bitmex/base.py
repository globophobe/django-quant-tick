import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
from pandas import DataFrame

from cryptofeed_werks.lib import candles_to_data_frame

from .api import format_bitmex_api_timestamp, get_bitmex_api_timestamp
from .candles import get_candles
from .constants import S3_URL, XBTUSD
from .trades import get_trades


class BitmexMixin:
    def get_uid(self, trade: dict) -> str:
        """Get uid."""
        return str(trade["trdMatchID"])

    def get_timestamp(self, trade: dict) -> datetime:
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
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> DataFrame:
        """Get candles from Exchange API."""
        # Timestamp is at candle close.
        ts_from = timestamp_from + pd.Timedelta(value="1t")
        candles = get_candles(
            self.symbol.api_symbol,
            ts_from,
            format_bitmex_api_timestamp(timestamp_to),
            bin_size="1m",
            log_format=f"{self.log_format} validating",
        )
        return candles_to_data_frame(timestamp_from, timestamp_to, candles)


class BitmexRESTMixin(BitmexMixin):
    def get_pagination_id(self, timestamp_to: datetime) -> str:
        """Get pagination_id."""
        return format_bitmex_api_timestamp(timestamp_to)

    def iter_api(self, timestamp_from: datetime, pagination_id: str):
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
    def get_url(self, date: datetime.date) -> str:
        """Get CSV file url."""
        date_string = date.strftime("%Y%m%d")
        return f"{S3_URL}{date_string}.csv.gz"

    def parse_dtypes_and_strip_columns(self, data_frame: DataFrame) -> DataFrame:
        """Parse dtypes and strip unnecessary columns."""
        data_frame["timestamp"] = pd.to_datetime(
            data_frame["timestamp"], format="%Y-%m-%dD%H:%M:%S.%f"
        )
        data_frame = data_frame.rename(columns={"trdMatchID": "uid"})
        # BitMEX XBTUSD size is volume. However, quanto contracts are not
        if self.symbol.api_symbol == XBTUSD:
            data_frame = data_frame.rename(columns={"foreignNotional": "volume"})
        else:
            raise NotImplementedError
        return super().parse_dtypes_and_strip_columns(data_frame)
