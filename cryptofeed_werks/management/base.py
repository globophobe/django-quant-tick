from typing import Optional

from django.core.management.base import BaseCommand, CommandParser

from cryptofeed_werks.constants import Exchange
from cryptofeed_werks.lib import parse_period_from_to
from cryptofeed_werks.models import Symbol


class BaseAggregatedTradeDataCommand(BaseCommand):
    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("exchange", type=Exchange, choices=Exchange.values)
        parser.add_argument("symbol")
        parser.add_argument("min-volume", type=int)
        parser.add_argument("--date-to", type=str, default=None)
        parser.add_argument("--time-to", type=str, default=None)
        parser.add_argument("--date-from", type=str, default=None)
        parser.add_argument("--time-from", type=str, default=None)

    def handle(self, *args, **options) -> Optional[dict]:
        exchange = options["exchange"]
        symbol = options["symbol"]
        min_volume = options["min-volume"]
        verbosity = options["verbosity"]
        try:
            symbol = Symbol.objects.get(
                exchange=exchange, api_symbol=symbol, min_volume=min_volume
            )
        except Symbol.DoesNotExist:
            print(f"{exchange} {symbol} {min_volume} not registered")
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
                "verbosity": verbosity,
            }
