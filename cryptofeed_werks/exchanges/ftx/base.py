from datetime import datetime
from decimal import Decimal

from pandas import DataFrame

from cryptofeed_werks.lib import candles_to_data_frame, timestamp_to_inclusive

from .api import format_ftx_api_timestamp
from .candles import get_candles
from .trades import get_ftx_trades_timestamp, get_trades


class FTXMixin:
    def get_pagination_id(self, timestamp_to: datetime) -> float:
        """Get pagination_id."""
        return format_ftx_api_timestamp(timestamp_to)

    def iter_api(self, timestamp_from: datetime, pagination_id: str) -> tuple:
        """Iterate Coinbase API."""
        return get_trades(
            self.symbol.api_symbol,
            timestamp_from,
            pagination_id,
            log_format=self.log_format,
        )

    def get_uid(self, trade: dict) -> datetime:
        """Get uid."""
        return str(trade["id"])

    def get_timestamp(self, trade: dict) -> datetime:
        """Get timestamp."""
        return get_ftx_trades_timestamp(trade)

    def get_nanoseconds(self, trade: dict) -> int:
        """Get nanoseconds."""
        return self.get_timestamp(trade).nanosecond

    def get_price(self, trade: dict) -> Decimal:
        """Get price."""
        return trade["price"]

    def get_volume(self, trade: dict) -> Decimal:
        """Get volume."""
        return self.get_price(trade) * self.get_notional(trade)

    def get_notional(self, trade: dict) -> Decimal:
        """Get notional."""
        return trade["size"]

    def get_tick_rule(self, trade: dict) -> int:
        """Get tick rule."""
        return 1 if trade["side"] == "buy" else -1

    def get_index(self, trade: dict) -> int:
        """Get index."""
        return int(trade["id"])

    def get_candles(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> DataFrame:
        """Get candles from Exchange API."""
        ts_to = timestamp_to_inclusive(timestamp_from, timestamp_to, value="1t")
        candles = get_candles(
            self.symbol.api_symbol,
            timestamp_from,
            format_ftx_api_timestamp(ts_to),
            resolution=60,
            log_format=f"{self.log_format} validating",
        )
        return candles_to_data_frame(timestamp_from, timestamp_to, candles)
