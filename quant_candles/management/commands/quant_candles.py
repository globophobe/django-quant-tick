import logging

from quant_candles.controllers import aggregate_candles, aggregate_trade_summary
from quant_candles.exchanges.api import api
from quant_candles.management.base import BaseQuantCandleCommand
from quant_candles.models import Candle, Symbol
from quant_candles.storage import convert_trade_data_to_hourly

logger = logging.getLogger(__name__)


class Command(BaseQuantCandleCommand):
    help = (
        "Get trades from exchange API or S3, "
        "aggregate trade data summary for symbol, "
        "convert trade data by minute to hourly, "
        "and create candles from trade data."
    )

    def handle(self, *args, **options) -> None:
        """Run command."""
        kwargs = super().handle(*args, **options)
        if kwargs:
            timestamp_from = kwargs["timestamp_from"]
            timestamp_to = kwargs["timestamp_to"]
            retry = kwargs["retry"]
            for exchange in kwargs["exchange"]:
                for symbol in Symbol.objects.filter(exchange=exchange):
                    logger.info(
                        "{symbol}: starting...".format(**{"symbol": str(symbol)})
                    )
                    api(symbol, timestamp_from, timestamp_to, retry)
                    convert_trade_data_to_hourly(symbol, timestamp_from, timestamp_to)
                    aggregate_trade_summary(symbol, timestamp_from, timestamp_to, retry)
            for candle in Candle.objects.filter(is_active=True):
                logger.info("{candle}: starting...".format(**{"candle": str(candle)}))
                aggregate_candles(candle, timestamp_from, timestamp_to, retry)
