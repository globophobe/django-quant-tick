import logging

from quant_candles.controllers import aggregate_trade_summary
from quant_candles.management.base import BaseTradeDataCommand

logger = logging.getLogger(__name__)


class Command(BaseTradeDataCommand):
    help = "Aggregate trade data summary for symbol."

    def handle(self, *args, **options) -> None:
        """Run command."""
        kwargs = super().handle(*args, **options)
        for k in kwargs:
            aggregate_trade_summary(**k)
