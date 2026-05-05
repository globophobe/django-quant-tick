from django.core.management.base import CommandError, CommandParser

from quant_tick.exchanges.api import exchange_candles
from quant_tick.management.base import BaseTradeDataWithRetryCommand


class Command(BaseTradeDataWithRetryCommand):
    help = "Get exchange candles from API."

    def get_queryset(self):
        return super().get_queryset().exclude(exchange_candle_resolution="")

    def add_arguments(self, parser: CommandParser) -> None:
        super().add_arguments(parser)
        parser.add_argument("--resolution", type=str, default=None)

    def handle(self, *args, **options) -> None:
        resolution_override = options["resolution"]
        for k in super().handle(*args, **options):
            symbol = k["symbol"]
            resolution = resolution_override or symbol.exchange_candle_resolution
            if not resolution:
                raise CommandError(
                    f"{symbol}: exchange candle resolution is required."
                )
            k["resolution"] = resolution
            exchange_candles(**k)
