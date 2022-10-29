from datetime import datetime
from decimal import Decimal

from quant_candles.controllers import IntegerPaginationMixin

from .api import get_bitflyer_api_timestamp, get_trades


class BitflyerMixin(IntegerPaginationMixin):
    def iter_api(self, timestamp_from: datetime, pagination_id: str) -> list:
        """Iterate Bitflyer API."""
        return get_trades(
            self.symbol.api_symbol, timestamp_from, pagination_id, self.log_format
        )

    def get_uid(self, trade: dict) -> str:
        """Get uid."""
        return str(trade["id"])

    def get_timestamp(self, trade: dict) -> datetime:
        """Get timestamp."""
        return get_bitflyer_api_timestamp(trade)

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
        return Decimal(trade["size"])

    def get_tick_rule(self, trade: dict) -> int:
        """Get tick rule."""
        return 1 if trade["side"] == "BUY" else -1

    def get_index(self, trade: dict) -> int:
        """Get index."""
        return int(trade["id"])
