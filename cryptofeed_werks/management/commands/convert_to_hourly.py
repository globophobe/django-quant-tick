from django.core.management.base import BaseCommand, CommandParser

from cryptofeed_werks.constants import Exchange
from cryptofeed_werks.lib import parse_period_from_to
from cryptofeed_werks.models import Symbol
from cryptofeed_werks.storage import convert_minute_to_hourly


class Command(BaseCommand):
    help = "Convert minute aggregated trade data to hourly."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("exchange", type=Exchange, choices=Exchange.values)
        parser.add_argument("symbol")
        parser.add_argument("--date-to", type=str, default=None)
        parser.add_argument("--time-to", type=str, default=None)
        parser.add_argument("--date-from", type=str, default=None)
        parser.add_argument("--time-from", type=str, default=None)

    def handle(self, *args, **options) -> tuple:
        exchange = options["exchange"]
        symbol = options["symbol"]
        verbose = bool(options["verbosity"])
        try:
            symbol = Symbol.objects.get(exchange=exchange, api_symbol=symbol)
        except Symbol.DoesNotExist:
            print(f"{exchange} {symbol} not registered.")
        else:
            timestamp_from, timestamp_to = parse_period_from_to(
                date_from=options["date_from"],
                time_from=options["time_from"],
                date_to=options["date_to"],
                time_to=options["time_to"],
            )
            convert_minute_to_hourly(
                symbol,
                timestamp_from=timestamp_from,
                timestamp_to=timestamp_to,
                verbose=verbose,
            )
