from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal

from quant_tick.constants import TradeDataRetry
from quant_tick.controllers import ExchangeREST
from quant_tick.lib import parse_datetime
from quant_tick.models import Symbol

from .candles import phoenix_candles
from .trades import get_trades, has_trades


def phoenix_trades(
    symbol: Symbol,
    timestamp_from: datetime,
    timestamp_to: datetime,
    on_data_frame: Callable,
    retry: TradeDataRetry = False,
    verbose: bool = False,
) -> None:
    PhoenixTrades(
        symbol,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        on_data_frame=on_data_frame,
        retry=retry,
        verbose=verbose,
    ).main()


class PhoenixTrades(ExchangeREST):
    """Partition-scoped REST fills; signed base quantity supplies taker direction.

    REST exposes no fill ID. Signature plus occurrence identifies rows within a
    complete partition, and index preserves the response order within a second.
    An empty backfill hour stops collection only if the venue has no older fills,
    including history before the requested window.
    """

    partition_scoped = True

    def is_history_exhausted(self, timestamp_to: datetime) -> bool:
        return not has_trades(
            self.symbol.api_symbol, datetime.fromtimestamp(0, UTC), timestamp_to
        )

    def get_pagination_id(self, timestamp_to: datetime) -> datetime:
        return timestamp_to

    def iter_api(self, timestamp_from: datetime, pagination_id: datetime) -> tuple:
        return get_trades(self.symbol.api_symbol, timestamp_from, pagination_id)

    def parse_data(self, data: list[dict]) -> list[dict]:
        parsed = []
        occurrences = Counter()
        for index, row in enumerate(data):
            price = Decimal(row["price"])
            base = Decimal(row["baseQty"])
            quote = Decimal(row["quoteQty"])
            if (
                not all(value.is_finite() for value in (price, base, quote))
                or price <= 0
                or base * quote >= 0
            ):
                raise ValueError("Phoenix fill has invalid price or signed quantities")
            signature = row["transactionSignature"]
            if not isinstance(signature, str) or not signature:
                raise ValueError("Phoenix fill lacks a transaction signature")
            occurrences[signature] += 1
            timestamp = parse_datetime(row["timestamp"])
            parsed.append(
                {
                    "uid": f"{signature}:{occurrences[signature]}",
                    "timestamp": timestamp,
                    "nanoseconds": timestamp.nanosecond,
                    "price": price,
                    "volume": abs(quote),
                    "notional": abs(base),
                    "tickRule": 1 if base > 0 else -1,
                    "index": len(data) - index - 1,
                }
            )
        return parsed

    def get_candles(self, timestamp_from: datetime, timestamp_to: datetime):
        return phoenix_candles(self.symbol.api_symbol, timestamp_from, timestamp_to)
