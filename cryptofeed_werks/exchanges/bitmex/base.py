import datetime
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from cryptofeed_werks.lib import get_min_time, get_next_time
from cryptofeed_werks.models import Candle

from .api import (
    format_bitmex_api_timestamp,
    get_active_futures,
    get_bitmex_api_timestamp,
    get_expired_futures,
    get_trades,
)
from .constants import S3_URL
from .lib import calculate_index


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


class BitmexRESTMixin(BitmexMixin):
    def get_pagination_id(self, data: Optional[dict] = None) -> str:
        """Get pagination_id."""
        return format_bitmex_api_timestamp(self.timestamp_to)

    def iter_api(self, timestamp_from, pagination_id):
        """Iterate Bitmex API."""
        return get_trades(
            self.symbol.name, timestamp_from, pagination_id, str(self.symbol)
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

    @property
    def get_columns(self) -> list:
        """Get CSV file columns."""
        return [
            "trdMatchID",
            "symbol",
            "timestamp",
            "price",
            "tickDirection",
            "foreignNotional",
        ]

    def parse_dataframe(self, data_frame: DataFrame) -> DataFrame:
        """Parse data_frame."""
        # No false positives.
        # Source: https://pandas.pydata.org/pandas-docs/stable/user_guide/
        # indexing.html#returning-a-view-versus-a-copy
        pd.options.mode.chained_assignment = None
        # Reset index.
        data_frame = calculate_index(data_frame)
        # Timestamp
        data_frame["timestamp"] = pd.to_datetime(
            data_frame["timestamp"], format="%Y-%m-%dD%H:%M:%S.%f"
        )
        # BitMEX XBTUSD size is volume. However, quanto contracts are not
        data_frame = data_frame.rename(
            columns={"trdMatchID": "uid", "foreignNotional": "volume"}
        )
        data_frame = calculate_index(data_frame)
        return super().parse_dataframe(data_frame)


class BitmexFuturesS3Mixin(BitmexS3Mixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symbols = self.get_symbols()

    def get_symbols(self) -> list:
        """Get symbols for root symbol."""
        active_futures = get_active_futures(
            self.symbol,
            date_from=self.period_from,
            date_to=self.period_to,
            verbose=self.verbose,
        )
        expired_futures = get_expired_futures(
            self.symbol,
            date_from=self.period_from,
            date_to=self.period_to,
            verbose=self.verbose,
        )
        return active_futures + expired_futures

    def has_symbols(self, data: dict) -> bool:
        """Is there data for all active symbols?"""
        return all([data.get(s["symbol"], None) for s in self.active_symbols])

    def get_symbol_data(self, symbol: str) -> dict:
        """Get data for symbol."""
        return [s for s in self.symbols if s["symbol"] == symbol][0]

    def has_data(self, date: datetime.date) -> bool:
        """Is there data for each active symbol?"""
        if not self.active_symbols:
            print(f"{self.log_format} No data")
            return True
        else:
            missing = Candle.objects.filter(
                symbol=self.symbol,
                futures__in=self.active_symbols,
                timestamp__gte=get_min_time(date),
                timestamp__lt=get_next_time(date),
                ok=False,
            )
            if self.has_symbols(daily_trades.data):
                date = date.isoformat()
                print(f"{self.log_format} {date} OK")
                return True

    def get_data(self, data_frame: DataFrame) -> dict:
        """Get data, including listing and expiry."""
        data = super().get_data(data_frame)
        for s in self.active_symbols:
            symbol = s["symbol"]
            # API data
            # Dataframe
            if symbol in data:
                d = self.get_symbol_data(symbol)
                listing = d["listing"].replace(tzinfo=datetime.timezone.utc)
                expiry = d["expiry"].replace(tzinfo=datetime.timezone.utc)
                data[symbol]["listing"] = listing
                data[symbol]["expiry"] = expiry
        return data
