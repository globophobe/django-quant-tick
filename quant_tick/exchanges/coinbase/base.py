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

    def parse_data(self, data: list) -> list:
        parsed = []
        for trade in data:
            timestamp = get_coinbase_trades_timestamp(trade)
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
                    "tickRule": 1 if trade["side"] == "sell" else -1,
                    "index": int(trade["trade_id"]),
                }
            )
        return parsed

    def get_candles(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> DataFrame:
        return coinbase_candles(
            self.symbol.api_symbol, timestamp_from, timestamp_to, granularity=60
        )
