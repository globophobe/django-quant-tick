from quant_tick.management.base import BaseTradeDataCommand
from quant_tick.storage import convert_trade_data_to_daily


class Command(BaseTradeDataCommand):
    help = "Convert trade data by minute, or hourly, to daily."

    def handle(self, *args, **options) -> None:
        kwargs = super().handle(*args, **options)
        for k in kwargs:
            convert_trade_data_to_daily(**k)
