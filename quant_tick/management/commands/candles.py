from quant_tick.management.base import BaseCandleCommand


class Command(BaseCandleCommand):
    help = "Create candles from trade data."

    def handle(self, *args, **options) -> None:
        for k in super().handle(*args, **options):
            k["candle"].candles(k["timestamp_from"], k["timestamp_to"], k["retry"])
