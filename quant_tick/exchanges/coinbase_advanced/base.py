from datetime import UTC, datetime
from decimal import Decimal

from pandas import DataFrame

from .trades import (
    get_coinbase_advanced_trades,
    get_coinbase_advanced_trades_timestamp,
)


class CoinbaseAdvancedMixin:
    """Coinbase Advanced mixin."""

    def get_pagination_id(self, timestamp_to: datetime) -> int:
        return int(timestamp_to.timestamp())

    def iter_api(self, timestamp_from: datetime, pagination_id: int) -> tuple:
        timestamp_to = datetime.fromtimestamp(pagination_id, tz=UTC)
        trades = get_coinbase_advanced_trades(
            self.symbol.api_symbol,
            timestamp_from,
            timestamp_to,
            log_format=self.log_format,
        )
        return trades, True, None

    def parse_data(self, data: list) -> list:
        parsed = []
        for trade in data:
            timestamp = get_coinbase_advanced_trades_timestamp(trade)
            price = Decimal(trade["price"])
            notional = Decimal(trade["size"])
            parsed.append(
                {
                    "uid": str(trade["trade_id"]),
                    "timestamp": timestamp,
                    "nanoseconds": timestamp.nanosecond,
                    "price": price,
                    "volume": price * notional,
                    "notional": notional,
                    "tickRule": 1 if trade["side"] == "SELL" else -1,
                    "index": int(trade["trade_id"]),
                }
            )
        return parsed

    def get_candles(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> DataFrame:
        return DataFrame([])
