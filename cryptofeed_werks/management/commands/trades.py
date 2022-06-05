from django.core.management.base import BaseCommand, CommandParser

from cryptofeed_werks.constants import Exchange
from cryptofeed_werks.exchanges.api import api
from cryptofeed_werks.lib import parse_period_from_to
from cryptofeed_werks.models import Symbol


class Command(BaseCommand):
    help = "Get trades from exchange API or S3"

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("exchange", type=Exchange, choices=Exchange.values)
        parser.add_argument("symbol")
        parser.add_argument("--date-to", type=str, default=None)
        parser.add_argument("--time-to", type=str, default=None)
        parser.add_argument("--date-from", type=str, default=None)
        parser.add_argument("--time-from", type=str, default=None)
        parser.add_argument("--retry", action="store_true")

    def handle(self, *args, **options) -> tuple:
        exchange = options["exchange"]
        symbol = options["symbol"]
        retry = options["retry"]
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
            api(symbol, timestamp_from, timestamp_to, retry, verbose)
