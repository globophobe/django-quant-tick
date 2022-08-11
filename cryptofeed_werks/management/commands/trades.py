from django.core.management.base import CommandParser

from cryptofeed_werks.exchanges.api import api
from cryptofeed_werks.management.base import BaseAggregatedTradeDataCommand


class Command(BaseAggregatedTradeDataCommand):
    help = "Get trades from exchange API or S3"

    def add_arguments(self, parser: CommandParser) -> None:
        super().add_arguments(parser)
        parser.add_argument("--retry", action="store_true")

    def handle(self, *args, **options) -> None:
        kwargs = super().handle(*args, **options)
        if kwargs:
            kwargs["retry"] = options["retry"]
            api(**kwargs)
