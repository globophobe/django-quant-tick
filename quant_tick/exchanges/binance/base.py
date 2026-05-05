import datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.constants import SymbolType
from quant_tick.controllers import SequentialIntegerMixin
from quant_tick.lib import set_type_decimal

from .candles import binance_candles
from .constants import FUTURES_S3_URL, SPOT_S3_URL
from .trades import get_binance_trades_timestamp, get_trades


class BinanceMixin(SequentialIntegerMixin):
    """Binance mixin."""

    def iter_api(self, timestamp_from: datetime.datetime, pagination_id: str) -> tuple:
        return get_trades(
            self.symbol.api_symbol,
            timestamp_from,
            pagination_id,
            symbol_type=self.symbol.symbol_type,
            log_format=self.log_format,
        )

    def parse_data(self, data: list) -> list:
        parsed = []
        for trade in data:
            timestamp = get_binance_trades_timestamp(trade)
            price = Decimal(trade["price"])
            notional = Decimal(trade["qty"])
            parsed.append(
                {
                    "uid": str(trade["id"]),
                    "timestamp": timestamp,
                    "nanoseconds": 0,
                    "price": price,
                    "volume": price * notional,
                    "notional": notional,
                    "tickRule": 1 if not trade["isBuyerMaker"] else -1,
                    "index": int(trade["id"]),
                }
            )
        return parsed

    def assert_data_frame(
        self,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
        data_frame: DataFrame,
        trades: list | None = None,
    ) -> None:
        """Assert Binance-specific integrity constraints."""
        # Duplicates.
        assert len(data_frame["uid"].unique()) == len(trades)
        # Missing orders.
        expected = len(trades) - 1
        diff = data_frame["index"].diff().dropna()
        assert abs(diff.sum()) == expected

    def get_candles(
        self, timestamp_from: datetime.datetime, timestamp_to: datetime.datetime
    ) -> DataFrame:
        return binance_candles(
            self.symbol.api_symbol,
            timestamp_from,
            timestamp_to,
            interval="1m",
            symbol_type=self.symbol.symbol_type,
            limit=60,
        )


class BinanceS3Mixin(BinanceMixin):
    """Binance S3 mixin."""

    @property
    def csv_columns(self) -> list:
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
        symbol = self.symbol.api_symbol
        date_str = date.strftime("%Y-%m-%d")
        s3_url = (
            FUTURES_S3_URL
            if self.symbol.symbol_type == SymbolType.PERPETUAL
            else SPOT_S3_URL
        )
        return f"{s3_url}/{symbol}/{symbol}-trades-{date_str}.zip"

    def parse_dtypes_and_strip_columns(self, df: DataFrame) -> DataFrame:
        """Parse Binance S3 columns into the canonical trade schema."""
        df = set_type_decimal(df, "price")
        df = set_type_decimal(df, "qty")
        # S3 files are daily, so first timestamp
        times = df["time"].astype("int64")
        first_time = int(times.iloc[0])
        # Milliseconds: ~13 digits (1e12), microseconds: ~16 digits (1e15)
        unit = "us" if first_time > 1e14 else "ms"
        df["timestamp"] = pd.to_datetime(times, unit=unit, utc=True)
        df["nanoseconds"] = 0
        if unit == "us":
            df["nanoseconds"] = (times % 1000) * 1000
        df = df.rename(columns={"id": "uid", "qty": "notional"})
        df["volume"] = df["price"] * df["notional"]
        df["tickRule"] = 1
        df.loc[df["isBuyerMaker"] == "True", "tickRule"] = -1
        return df[self.columns]
