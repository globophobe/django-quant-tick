from quant_tick.exchanges.order_book.storage import convert_order_book_data_to_daily
from quant_tick.lib import get_current_time
from quant_tick.management.base import BaseTradeDataCommand
from quant_tick.storage import get_compact_max_timestamp_to


class Command(BaseTradeDataCommand):
    help = "Compact order book minute summaries into hourly and daily Parquet files."

    def handle(self, *args, **options):
        max_timestamp_to = get_compact_max_timestamp_to(get_current_time())
        for kwargs in super().handle(*args, **options):
            kwargs["timestamp_to"] = min(kwargs["timestamp_to"], max_timestamp_to)
            if kwargs["timestamp_from"] < kwargs["timestamp_to"]:
                convert_order_book_data_to_daily(**kwargs)
