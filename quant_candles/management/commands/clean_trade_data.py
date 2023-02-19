from quant_candles.management.base import BaseTradeDataCommand
from quant_candles.storage import (
    clean_trade_data_with_non_existing_files,
    clean_unlinked_trade_data_files,
)


class Command(BaseTradeDataCommand):
    help = "Clean storage, and decrease storage frequency."

    def handle(self, *args, **options) -> None:
        """Run command."""
        kwargs = super().handle(*args, **options)
        for k in kwargs:
            clean_trade_data_with_non_existing_files(**k)
            clean_unlinked_trade_data_files(**k)
