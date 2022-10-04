from quant_werks.management.base import BaseAggregatedTradeDataCommand
from quant_werks.storage import convert_aggregated_to_hourly


class Command(BaseAggregatedTradeDataCommand):
    help = (
        "Convert trade data aggregated by minute to hourly, to reduce file operations."
    )

    def handle(self, *args, **options) -> None:
        kwargs = super().handle(*args, **options)
        if kwargs:
            convert_aggregated_to_hourly(**kwargs)
