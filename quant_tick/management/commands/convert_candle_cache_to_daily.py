from quant_tick.management.base import BaseCandleCommand
from quant_tick.storage import convert_candle_cache_to_daily


class Command(BaseCandleCommand):
    """Convert candle cache to daily."""

    help = "Convert candle cache by minute, or hourly, to daily."

    def handle(self, *args, **options) -> None:
        """Run command."""
        kwargs = super().handle(*args, **options)
        for k in kwargs:
            convert_candle_cache_to_daily(k["candle"])
