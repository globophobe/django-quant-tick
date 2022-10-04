import logging
from typing import Optional

from django.core.management.base import BaseCommand, CommandParser

from quant_werks.constants import Exchange
from quant_werks.lib import parse_period_from_to
from quant_werks.models import Symbol

logger = logging.getLogger(__name__)


class BaseAggregatedTradeDataCommand(BaseCommand):
    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("exchange", type=Exchange, choices=Exchange.values)
        parser.add_argument("symbol")
        parser.add_argument("--aggregate", type=bool)
        parser.add_argument("--date-to", type=str, default=None)
        parser.add_argument("--time-to", type=str, default=None)
        parser.add_argument("--date-from", type=str, default=None)
        parser.add_argument("--time-from", type=str, default=None)
        parser.add_argument("--filter", type=int, default=0)

    @classmethod
    def get_symbol_display(
        cls,
        exchange: str,
        symbol: str,
        should_aggregate_trades: bool,
        significant_trade_filter: int,
    ) -> str:
        """Get symbol display."""
        parts = [exchange, symbol]
        if should_aggregate_trades:
            parts += ["aggregated", str(significant_trade_filter)]
        else:
            parts.append("raw")
        return " ".join(parts)

    def handle(self, *args, **options) -> Optional[dict]:
        exchange = options["exchange"]
        symbol = options["symbol"]
        should_aggregate_trades = options["aggregate"]
        significant_trade_filter = options["filter"]
        try:
            symbol = Symbol.objects.get(
                exchange=exchange,
                api_symbol=symbol,
                should_aggregate_trades=should_aggregate_trades,
                significant_trade_filter=significant_trade_filter,
            )
        except Symbol.DoesNotExist:
            s = self.get_symbol_display(
                exchange, symbol, should_aggregate_trades, significant_trade_filter
            )
            logger.warn(f"{s} not registered")
        else:
            timestamp_from, timestamp_to = parse_period_from_to(
                date_from=options["date_from"],
                time_from=options["time_from"],
                date_to=options["date_to"],
                time_to=options["time_to"],
            )
            return {
                "symbol": symbol,
                "timestamp_from": timestamp_from,
                "timestamp_to": timestamp_to,
            }
