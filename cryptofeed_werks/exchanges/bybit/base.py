from datetime import datetime
from decimal import Decimal

import httpx
import pandas as pd

from cryptofeed_werks.controllers import SequentialIntegerMixin

from .api import get_bybit_api_timestamp, get_trades
from .constants import MAX_RESULTS, S3_URL


class BybitMixin:
    """Bybit mixin."""

    def get_uid(self, trade: dict) -> str:
        """Get uid."""
        return str(trade["id"])

    def get_timestamp(self, trade: dict) -> datetime:
        """Get timestamp."""
        return get_bybit_api_timestamp(trade)

    def get_nanoseconds(self, trade: dict) -> int:
        """Get nanoseconds."""
        return self.get_timestamp(trade).nanosecond

    def get_price(self, trade: dict) -> Decimal:
        """Get price."""
        return Decimal(trade["price"])

    def get_volume(self, trade: dict) -> Decimal:
        """Get volume."""
        return Decimal(trade["qty"])

    def get_notional(self, trade: dict) -> Decimal:
        """Get notional."""
        return self.get_volume(trade) / self.get_price(trade)

    def get_tick_rule(self, trade):
        """Get tick rule."""
        return 1 if trade["side"] == "Buy" else -1

    def get_index(self, trade: dict) -> int:
        """Get index."""
        return trade["id"]


class BybitRESTMixin(SequentialIntegerMixin, BybitMixin):
    """Bybit REST mixin."""

    def get_pagination_id(self, data=None):
        """Get pagination_id."""
        pagination_id = super().get_pagination_id(data=data)
        # Bybit pagination is donkey balls
        if pagination_id is not None:
            pagination_id = pagination_id - MAX_RESULTS
            assert pagination_id > 0
        return pagination_id

    def iter_api(self, symbol, pagination_id, log_format):
        """Iterate API."""
        return get_trades(symbol, self.timestamp_from, pagination_id, log_format)


class BybitS3Mixin(BybitMixin):
    """Bybit S3 mixin."""

    def get_url(self, date):
        """Get URL."""
        directory = f"{S3_URL}{self.symbol}/"
        response = httpx.get(directory)
        if response.status_code == 200:
            return f"{S3_URL}{self.symbol}/{self.symbol}{date.isoformat()}.csv.gz"
        else:
            print(f"{self.exchange.capitalize()} {self.symbol}: No data")

    @property
    def get_columns(self):
        """Get columns."""
        return ("trdMatchID", "timestamp", "price", "size", "tickDirection")

    def parse_dataframe(self, data_frame):
        """Parse dataframe."""
        # No false positives.
        # Source: https://pandas.pydata.org/pandas-docs/stable/user_guide/
        # indexing.html#returning-a-view-versus-a-copy
        pd.options.mode.chained_assignment = None
        # Bybit is reversed.
        data_frame = data_frame.iloc[::-1]
        data_frame["index"] = data_frame.index.values[::-1]
        data_frame["timestamp"] = pd.to_datetime(data_frame["timestamp"], unit="s")
        data_frame = data_frame.rename(columns={"trdMatchID": "uid", "size": "volume"})
        return super().parse_dataframe(data_frame)
