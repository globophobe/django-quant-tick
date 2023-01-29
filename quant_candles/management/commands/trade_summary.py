import logging

from quant_candles.controllers import aggregate_trade_summary
from quant_candles.management.base import BaseQuantCandleCommand
from quant_candles.models import Symbol

logger = logging.getLogger(__name__)


class Command(BaseQuantCandleCommand):
    help = "Aggregate trade data summary for symbol."

    def handle(self, *args, **options) -> None:
        """Run command."""
        kwargs = super().handle(*args, **options)
        timestamp_from = kwargs["timestamp_from"]
        timestamp_to = kwargs["timestamp_to"]
        retry = kwargs["retry"]
        for exchange in kwargs["exchange"]:
            for symbol in Symbol.objects.filter(exchange=exchange):
                logger.info("{symbol}: starting...".format(**{"symbol": str(symbol)}))
                aggregate_trade_summary(symbol, timestamp_from, timestamp_to, retry)
