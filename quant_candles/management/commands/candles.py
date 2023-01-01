import logging

from django.core.management.base import BaseCommand, CommandParser

from quant_candles.lib import parse_period_from_to
from quant_candles.models import Candle

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Create candles from trade data"

    def add_arguments(self, parser: CommandParser) -> None:
        super().add_arguments(parser)
        parser.add_argument(
            "name",
            type=str,
            choices=Candle.objects.all().values_list("code_name", flat=True),
        )
        parser.add_argument("--date-to", type=str, default=None)
        parser.add_argument("--time-to", type=str, default=None)
        parser.add_argument("--date-from", type=str, default=None)
        parser.add_argument("--time-from", type=str, default=None)
        parser.add_argument("--retry", action="store_true")

    def handle(self, *args, **options) -> None:
        name = options["name"]
        try:
            candle = Candle.objects.get(code_name=name)
        except Candle.DoesNotExist:
            logger.warn(f"{name} not registered")
        else:
            timestamp_from, timestamp_to = parse_period_from_to(
                date_from=options["date_from"],
                time_from=options["time_from"],
                date_to=options["date_to"],
                time_to=options["time_to"],
            )
