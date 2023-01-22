import logging

from django.core.management.base import BaseCommand, CommandParser

from quant_candles.constants import Exchange
from quant_candles.controllers import aggregate_candles, aggregate_trade_summary
from quant_candles.exchanges.api import api
from quant_candles.models import Candle, Symbol
from quant_candles.serializers import QuantCandleParameterSerializer
from quant_candles.storage import convert_trade_data_to_hourly

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = (
        "Get trades from exchange API or S3, "
        "aggregate trade data summary for symbol, "
        "convert trade data by minute to hourly, "
        "and create candles from trade data."
    )

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        super().add_arguments(parser)
        parser.add_argument(
            "--exchange", type=Exchange, default=Exchange.values, nargs="+"
        )
        parser.add_argument(
            "--time-ago", type=str, default="3d", help="5t, 12h, 1d, 1w, etc."
        )
        parser.add_argument("--retry", action="store_true")

    def handle(self, *args, **options) -> None:
        """Run command."""
        serializer = QuantCandleParameterSerializer(data=options)
        serializer.is_valid()
        data = serializer.validated_data
        timestamp_from = data["timestamp_from"]
        timestamp_to = data["timestamp_to"]
        retry = data["retry"]
        for exchange in options["exchange"]:
            for symbol in Symbol.objects.filter(exchange=exchange):
                logger.info("{symbol}: starting...".format(**{"symbol": str(symbol)}))
                api(symbol, timestamp_from, timestamp_to, retry)
                convert_trade_data_to_hourly(symbol, timestamp_from, timestamp_to)
                aggregate_trade_summary(symbol, timestamp_from, timestamp_to, retry)
        for candle in Candle.objects.filter(is_active=True):
            logger.info("{candle}: starting...".format(**{"candle": str(candle)}))
            aggregate_candles(candle, timestamp_from, timestamp_to, retry)
