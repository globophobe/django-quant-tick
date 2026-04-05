from datetime import datetime
from decimal import Decimal

from pandas import DataFrame

from quant_tick.controllers import IntegerPaginationMixin

from .candles import coinbase_candles
from .trades import get_coinbase_trades_timestamp, get_trades


class CoinbaseMixin(IntegerPaginationMixin):
    """Coinbase mixin."""

    def iter_api(self, timestamp_from: datetime, pagination_id: str) -> tuple:
        return get_trades(
            self.symbol.api_symbol,
            timestamp_from,
            pagination_id,
            log_format=self.log_format,
        )

    def get_uid(self, trade: dict) -> str:
        return str(trade["trade_id"])

    def get_timestamp(self, trade: dict) -> datetime:
        return get_coinbase_trades_timestamp(trade)

    def get_nanoseconds(self, trade: dict) -> int:
        return self.get_timestamp(trade).nanosecond

    def get_price(self, trade: dict) -> Decimal:
        return Decimal(trade["price"])

    def get_volume(self, trade: dict) -> Decimal:
        return self.get_price(trade) * self.get_notional(trade)

    def get_notional(self, trade: dict) -> Decimal:
        return Decimal(trade["size"])

    def get_tick_rule(self, trade: dict) -> int:
        """Get tick rule.

        Buy side indicates a down-tick because the maker was a buy order and
        their order was removed. Conversely, sell side indicates an up-tick.
        """
        return 1 if trade["side"] == "sell" else -1

    def get_index(self, trade: dict) -> int:
        return int(trade["trade_id"])

    def get_candles(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> DataFrame:
        return coinbase_candles(
            self.symbol.api_symbol, timestamp_from, timestamp_to, granularity=60
        )
