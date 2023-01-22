from django.core.management.base import CommandParser

from quant_candles.controllers import aggregate_trade_summary
from quant_candles.management.base import BaseTradeDataCommand


class Command(BaseTradeDataCommand):
    help = "Aggregate trade data summary for symbol."

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        super().add_arguments(parser)
        parser.add_argument("--retry", action="store_true")

    def handle(self, *args, **options) -> None:
        """Run command."""
        kwargs = super().handle(*args, **options)
        if kwargs:
            kwargs["retry"] = options["retry"]
            aggregate_trade_summary(**kwargs)
