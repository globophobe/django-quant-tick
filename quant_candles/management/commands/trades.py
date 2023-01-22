from django.core.management.base import CommandParser

from quant_candles.exchanges.api import api
from quant_candles.management.base import BaseTradeDataCommand


class Command(BaseTradeDataCommand):
    help = "Get trades from exchange API or S3."

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        super().add_arguments(parser)
        parser.add_argument("--retry", action="store_true")

    def handle(self, *args, **options) -> None:
        """Run command."""
        kwargs = super().handle(*args, **options)
        if kwargs:
            kwargs["retry"] = options["retry"]
            api(**kwargs)
