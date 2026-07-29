import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
from pandas import DataFrame

from quant_tick.constants import SymbolType

from .candles import bybit_candles, get_bybit_category
from .constants import DERIVATIVES_S3_URL, SPOT_S3_URL

NANOSECONDS_PER_SECOND = 1_000_000_000


def _archive_timestamp_nanoseconds(values: pd.Series) -> pd.Series:
    strings = values.astype("string")
    parts = strings.str.partition(".")
    seconds = pd.to_numeric(parts[0], errors="raise").astype("int64")
    fractions = parts[2].str[:9].str.pad(9, side="right", fillchar="0")
    fractions = pd.to_numeric(
        fractions.mask(fractions == "", "0"),
        errors="raise",
    ).astype("int64")
    return seconds * NANOSECONDS_PER_SECOND + fractions



class BybitMixin:
    """Bybit mixin."""

    def get_candles(
        self,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
    ) -> DataFrame:
        return bybit_candles(
            self.symbol.api_symbol,
            timestamp_from,
            timestamp_to,
            resolution="1m",
            category=get_bybit_category(self.symbol.exchange),
        )


class BybitS3Mixin(BybitMixin):
    """Bybit S3 mixin."""

    @property
    def gzipped_csv_columns(self) -> list[str]:
        return [
            "timestamp",
            "symbol",
            "side",
            "size",
            "price",
            "trdMatchID",
        ]

    def prepare_archive_chunk(self, data_frame: DataFrame) -> DataFrame:
        symbols = data_frame.pop("symbol")
        if not symbols.eq(self.symbol.api_symbol).all():
            raise ValueError(
                f"Bybit archive contains rows outside {self.symbol.api_symbol}"
            )
        return data_frame

    def get_archive_timestamp_nanoseconds(
        self,
        data_frame: DataFrame,
    ) -> pd.Series:
        return _archive_timestamp_nanoseconds(data_frame["timestamp"])

    def get_url(self, date: datetime.date) -> str:
        symbol = self.symbol.api_symbol
        return f"{DERIVATIVES_S3_URL}/{symbol}/{symbol}{date.isoformat()}.csv.gz"

    def parse_dtypes_and_strip_columns(self, data_frame: DataFrame) -> DataFrame:
        """Parse Bybit S3 columns into the canonical trade schema."""
        df = data_frame
        total_nanoseconds = _archive_timestamp_nanoseconds(df["timestamp"])
        df["timestamp"] = pd.to_datetime(
            total_nanoseconds // 1000,
            unit="us",
            utc=True,
        )
        df["nanoseconds"] = (total_nanoseconds % 1000).astype("int64")
        df["price"] = df["price"].map(Decimal)
        size = df["size"].map(Decimal)
        category = get_bybit_category(self.symbol.exchange)
        if category == "inverse":
            df["volume"] = size
            df["notional"] = size / df["price"]
        else:
            df["notional"] = size
            df["volume"] = size * df["price"]
        df["tickRule"] = np.where(
            df["side"].str.lower() == "buy",
            1,
            -1,
        )
        df = df.rename(columns={"trdMatchID": "uid"})
        return df[self.columns].reset_index(drop=True)


class BybitSpotS3Mixin(BybitMixin):
    """Bybit spot S3 mixin."""

    @property
    def gzipped_csv_columns(self) -> list[str]:
        return ["id", "timestamp", "price", "volume", "side"]

    def get_url(self, date: datetime.date) -> str:
        symbol = self.symbol.api_symbol
        return f"{SPOT_S3_URL}/{symbol}/{symbol}_{date.isoformat()}.csv.gz"

    def get_archive_timestamp_nanoseconds(
        self,
        data_frame: DataFrame,
    ) -> pd.Series:
        return data_frame["timestamp"].astype("int64") * 1_000_000

    def parse_dtypes_and_strip_columns(self, data_frame: DataFrame) -> DataFrame:
        """Parse Bybit spot S3 columns into the canonical trade schema."""
        df = data_frame.copy()
        df["uid"] = df["id"]
        df["timestamp"] = pd.to_datetime(
            df["timestamp"].astype("int64"),
            unit="ms",
            utc=True,
        )
        df["nanoseconds"] = 0
        df["price"] = df["price"].map(Decimal)
        size = df["volume"].map(Decimal)
        df["notional"] = size
        df["volume"] = size * df["price"]
        df["tickRule"] = np.where(
            df["side"].str.lower() == "buy",
            1,
            -1,
        )
        return df[self.columns].reset_index(drop=True)


def validate_bybit_trade_symbol(symbol) -> str:
    category = get_bybit_category(symbol.exchange)
    expected_type = (
        SymbolType.SPOT if category == "spot" else SymbolType.PERPETUAL
    )
    if symbol.symbol_type != expected_type:
        raise ValueError(f"{symbol.exchange} must use {expected_type} symbols.")
    return category
