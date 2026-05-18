from quant_tick.management.base import BaseTradeDataCommand
from quant_tick.lib import get_current_time
from quant_tick.storage import convert_trade_data_to_daily, get_compact_max_timestamp_to


class Command(BaseTradeDataCommand):
    help = "Convert trade data by minute, or hourly, to daily."

    def handle(self, *args, **options) -> None:
        kwargs = super().handle(*args, **options)
        max_timestamp_to = get_compact_max_timestamp_to(get_current_time())
        for k in kwargs:
            k["timestamp_to"] = min(k["timestamp_to"], max_timestamp_to)
            if k["timestamp_from"] >= k["timestamp_to"]:
                continue
            convert_trade_data_to_daily(**k)
