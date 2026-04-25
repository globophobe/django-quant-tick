from quant_tick.management.base import BaseTradeDataCommand
from quant_tick.storage import (
    clean_trade_data_overlaps,
    clean_trade_data_with_non_existing_files,
    clean_unlinked_trade_data_files,
)


class Command(BaseTradeDataCommand):
    help = "Clean storage, and decrease storage frequency."

    def handle(self, *args, **options) -> None:
        kwargs = super().handle(*args, **options)
        for k in kwargs:
            clean_trade_data_overlaps(**k)
            clean_trade_data_with_non_existing_files(**k)
            clean_unlinked_trade_data_files(**k)
