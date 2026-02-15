import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
from pandas import DataFrame

from quant_tick.controllers import SequentialIntegerMixin
from quant_tick.lib import set_dtypes

from .candles import binance_candles
from .constants import S3_URL
from .trades import get_binance_trades_timestamp, get_trades


class BinanceMixin(SequentialIntegerMixin):
    """Binance mixin."""

    def iter_api(self, timestamp_from: datetime.datetime, pagination_id: str) -> tuple:
        """Iterate Binance API."""
        return get_trades(
            self.symbol.api_symbol,
            timestamp_from,
            pagination_id,
            log_format=self.log_format,
        )

    def get_uid(self, trade: dict) -> str:
        """Get uid."""
        return str(trade["id"])

    def get_timestamp(self, trade: dict) -> datetime.datetime:
        """Get timestamp."""
        return get_binance_trades_timestamp(trade)

    def get_nanoseconds(self, trade: dict) -> int:
        """Get nanoseconds."""
        return self.get_timestamp(trade).nanosecond

    def get_price(self, trade: dict) -> Decimal:
        """Get price."""
        return Decimal(trade["price"])

    def get_volume(self, trade: dict) -> Decimal:
        """Get volume."""
        return self.get_price(trade) * self.get_notional(trade)

    def get_notional(self, trade: dict) -> Decimal:
        """Get notional."""
        return Decimal(trade["qty"])

    def get_tick_rule(self, trade: dict) -> int:
        """Get tick rule.

        If isBuyerMaker is true, order was filled by sell order
        """
        return 1 if not trade["isBuyerMaker"] else -1

    def get_index(self, trade: dict) -> int:
        """Get index."""
        return int(trade["id"])

    def assert_data_frame(
        self,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
        data_frame: DataFrame,
        trades: list | None = None,
    ) -> None:
        """Assertions on data_frame."""
        # Duplicates.
        assert len(data_frame["uid"].unique()) == len(trades)
        # Missing orders.
        expected = len(trades) - 1
        diff = data_frame["index"].diff().dropna()
        assert abs(diff.sum()) == expected

    def get_candles(
        self, timestamp_from: datetime.datetime, timestamp_to: datetime.datetime
    ) -> DataFrame:
        """Get candles from Exchange API."""
        return binance_candles(
            self.symbol.api_symbol,
            timestamp_from,
            timestamp_to,
            interval="1m",
            limit=60,
        )


class BinanceS3Mixin(BinanceMixin):
    """Binance S3 mixin."""

    @property
    def csv_columns(self) -> list:
        """CSV columns."""
        return [
            "id",
            "price",
            "qty",
            "quoteQty",
            "time",
            "isBuyerMaker",
            "isBestMatch",
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

    def get_url(self, date: datetime.date) -> str:
        """Get URL."""
        symbol = self.symbol.api_symbol
        date_str = date.strftime("%Y-%m-%d")
        return f"{S3_URL}/{symbol}/{symbol}-trades-{date_str}.zip"

    def get_index(self, trade: dict) -> int:
        """Get index.

        * No sequential index.
        """
        return np.nan

    def parse_dtypes_and_strip_columns(self, df: DataFrame) -> DataFrame:
        """Parse dtypes and strip columns."""
        df = df.copy()
        df = set_dtypes(df)
        # S3 files are daily, so first timestamp
        first_time = int(df["time"].iloc[0])
        # Milliseconds: ~13 digits (1e12), microseconds: ~16 digits (1e15)
        unit = "us" if first_time > 1e14 else "ms"
        df["timestamp"] = pd.to_datetime(
            df["time"].astype("int64"), unit=unit, utc=True
        )
        df["nanoseconds"] = 0
        if unit == "us":
            df["nanoseconds"] = (df["time"].astype("int64") % 1000) * 1000
        df = df.rename(columns={"id": "uid", "qty": "notional"})
        df["volume"] = df["price"] * df["notional"]
        df["tickRule"] = df["isBuyerMaker"].apply(lambda x: -1 if x == "True" else 1)
        return df[self.columns]
