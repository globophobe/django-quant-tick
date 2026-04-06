from quant_tick.exchanges.api import api
from quant_tick.management.base import BaseTradeDataWithRetryCommand


class Command(BaseTradeDataWithRetryCommand):
    help = "Get trades from exchange API or S3."

    def handle(self, *args, **options) -> None:
        kwargs = super().handle(*args, **options)
        for k in kwargs:
            api(**k)
