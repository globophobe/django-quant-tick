import datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.controllers import SequentialIntegerMixin
from quant_tick.lib import set_type_decimal

from .candles import binance_futures_candles
from .constants import S3_URL
from .trades import (
    get_binance_futures_trade_timestamp,
    get_binance_futures_trades,
)


class BinanceFuturesMixin(SequentialIntegerMixin):
    """Binance Futures mixin."""

    @property
    def columns(self) -> list[str]:
        return [
            "uid",
            "timestamp",
            "nanoseconds",
            "price",
            "volume",
            "notional",
            "tickRule",
            "ticks",
            "index",
        ]

    def iter_api(
        self,
        timestamp_from: datetime.datetime,
        pagination_id: int | None,
    ) -> tuple:
        return get_binance_futures_trades(
            self.symbol.api_symbol,
            timestamp_from,
            pagination_id,
            log_format=self.log_format,
        )

    def parse_data(self, data: list) -> list:
        parsed = []
        for trade in data:
            price = Decimal(trade["p"])
            notional = Decimal(trade["q"])
            aggregate_id = int(trade["a"])
            parsed.append(
                {
                    "uid": str(aggregate_id),
                    "timestamp": get_binance_futures_trade_timestamp(trade),
                    "nanoseconds": 0,
                    "price": price,
                    "volume": price * notional,
                    "notional": notional,
                    "ticks": int(trade["l"]) - int(trade["f"]) + 1,
                    "tickRule": -1 if trade["m"] else 1,
                    "index": aggregate_id,
                }
            )
        return parsed

    def get_candles(
        self,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
    ) -> DataFrame:
        return binance_futures_candles(
            self.symbol.api_symbol,
            timestamp_from,
            timestamp_to,
            interval="1m",
            limit=60,
        )


class BinanceFuturesS3Mixin(BinanceFuturesMixin):
    """Binance Futures S3 mixin."""

    @property
    def csv_columns(self) -> list[str]:
        return [
            "agg_trade_id",
            "price",
            "quantity",
            "first_trade_id",
            "last_trade_id",
            "transact_time",
            "is_buyer_maker",
        ]

    @property
    def columns(self) -> list[str]:
        return [
            "uid",
            "timestamp",
            "nanoseconds",
            "price",
            "volume",
            "notional",
            "tickRule",
            "ticks",
        ]

    def prepare_archive_chunk(self, df: DataFrame) -> DataFrame:
        header_rows = (
            df["agg_trade_id"].astype(str).str.lower() == "agg_trade_id"
        )
        return df.loc[~header_rows].copy() if header_rows.any() else df

    @staticmethod
    def get_archive_timestamp_unit(times: pd.Series) -> str:
        return "us" if int(times.iloc[0]) > 1e14 else "ms"

    def get_archive_timestamp_nanoseconds(self, df: DataFrame) -> pd.Series:
        times = df["transact_time"].astype("int64")
        scale = 1_000 if self.get_archive_timestamp_unit(times) == "us" else 1_000_000
        return times * scale

    def get_url(self, date: datetime.date) -> str:
        symbol = self.symbol.api_symbol
        date_str = date.isoformat()
        return f"{S3_URL}/{symbol}/{symbol}-aggTrades-{date_str}.zip"

    def parse_dtypes_and_strip_columns(self, data_frame: DataFrame) -> DataFrame:
        df = self.prepare_archive_chunk(data_frame)
        df = set_type_decimal(df, "price")
        df = set_type_decimal(df, "quantity")
        times = df["transact_time"].astype("int64")
        unit = self.get_archive_timestamp_unit(times)
        df["timestamp"] = pd.to_datetime(times, unit=unit, utc=True)
        df["nanoseconds"] = 0
        df = df.rename(columns={"agg_trade_id": "uid", "quantity": "notional"})
        df["volume"] = df["price"] * df["notional"]
        df["tickRule"] = 1
        is_buyer_maker = df["is_buyer_maker"].astype(str).str.lower() == "true"
        df.loc[is_buyer_maker, "tickRule"] = -1
        df["ticks"] = (
            df["last_trade_id"].astype("int64")
            - df["first_trade_id"].astype("int64")
            + 1
        )
        return df[self.columns]
