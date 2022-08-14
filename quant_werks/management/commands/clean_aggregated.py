from quant_werks.management.base import BaseAggregatedTradeDataCommand
from quant_werks.storage import clean_aggregated_storage


class Command(BaseAggregatedTradeDataCommand):
    help = "Clean storage, and decrease storage frequency."

    def handle(self, *args, **options) -> None:
        kwargs = super().handle(*args, **options)
        if kwargs:
            clean_aggregated_storage(**kwargs)
