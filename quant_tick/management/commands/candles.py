import logging

from quant_tick.controllers import aggregate_candles
from quant_tick.management.base import BaseCandleCommand

logger = logging.getLogger(__name__)


class Command(BaseCandleCommand):
    """Candles."""

    help = "Create candles from trade data."

    def handle(self, *args, **options) -> None:
        """Run command."""
        kwargs = super().handle(*args, **options)
        for k in kwargs:
            aggregate_candles(**k)
