from datetime import UTC, datetime
from decimal import Decimal

from pandas import DataFrame

from quant_tick.constants import SymbolType
from quant_tick.controllers import SequentialIntegerMixin
from quant_tick.models import TradeData

from .api import to_millis
from .candles import deribit_candles
from .trades import get_trades


class DeribitMixin(SequentialIntegerMixin):
    """Normalize Deribit spot and perpetual trades into DQT units."""

    def get_pagination_id(self, timestamp_to: datetime) -> dict[str, int]:
        uid = TradeData.objects.get_last_uid(self.symbol, timestamp_to)
        if uid is not None:
            return {"end_seq": int(uid) - 1}
        return {"end_timestamp": to_millis(timestamp_to) - 1}

    def iter_api(
        self,
        timestamp_from: datetime,
        pagination_id: dict[str, int],
    ) -> tuple[list[dict], bool, dict[str, int] | None]:
        return get_trades(
            self.symbol.api_symbol,
            timestamp_from,
            pagination_id,
            log_format=self.log_format,
        )

    def parse_data(self, data: list[dict]) -> list[dict]:
        parsed = []
        for trade in data:
            price = Decimal(str(trade["price"]))
            amount = Decimal(str(trade["amount"]))
            direction = trade["direction"]
            if direction not in {"buy", "sell"}:
                raise ValueError(f"Unexpected Deribit trade direction: {direction}")
            sequence = int(trade["trade_seq"])
            amount_is_base = (
                self.symbol.symbol_type == SymbolType.SPOT
                or "_" in self.symbol.api_symbol
            )
            volume = price * amount if amount_is_base else amount
            notional = amount if amount_is_base else amount / price
            parsed.append(
                {
                    "uid": str(sequence),
                    "timestamp": datetime.fromtimestamp(
                        int(trade["timestamp"]) / 1000,
                        tz=UTC,
                    ),
                    "nanoseconds": 0,
                    "price": price,
                    "volume": volume,
                    "notional": notional,
                    "tickRule": 1 if direction == "buy" else -1,
                    "index": sequence,
                }
            )
        return parsed

    def get_candles(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
    ) -> DataFrame:
        return deribit_candles(
            self.symbol.api_symbol,
            timestamp_from,
            timestamp_to,
            resolution="1m",
        )
