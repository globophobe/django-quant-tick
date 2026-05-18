from django.core.management.base import CommandParser

from quant_tick.constants import RETRY_INDETERMINATE
from quant_tick.exchanges.api import api
from quant_tick.management.base import BaseTradeDataWithRetryCommand


class Command(BaseTradeDataWithRetryCommand):
    help = "Get trades from exchange API or S3."

    def add_arguments(self, parser: CommandParser) -> None:
        super().add_arguments(parser)
        parser.add_argument(
            "--retry-indeterminate",
            action="store_true",
            help="Retry trade data where validation did not produce a result.",
        )

    def handle(self, *args, **options) -> None:
        retry_indeterminate = options["retry_indeterminate"]
        kwargs = super().handle(*args, **options)
        for k in kwargs:
            if retry_indeterminate:
                k["retry"] = RETRY_INDETERMINATE
            api(**k)
