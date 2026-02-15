from quant_tick.management.base import BaseCandleCommand


class Command(BaseCandleCommand):
    """Candles."""

    help = "Create candles from trade data."

    def handle(self, *args, **options) -> None:
        """Run command."""
        for k in super().handle(*args, **options):
            k["candle"].candles(k["timestamp_from"], k["timestamp_to"], k["retry"])
