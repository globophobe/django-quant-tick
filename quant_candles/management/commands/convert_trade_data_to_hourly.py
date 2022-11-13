from quant_candles.management.base import BaseTradeDataCommand
from quant_candles.storage import convert_trade_data_to_hourly


class Command(BaseTradeDataCommand):
    help = "Convert trade data by minute to hourly, to reduce file operations."

    def handle(self, *args, **options) -> None:
        kwargs = super().handle(*args, **options)
        if kwargs:
            convert_trade_data_to_hourly(**kwargs)