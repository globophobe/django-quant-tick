from quant_tick.constants import SymbolType
from quant_tick.exchanges.perpetual_stats import (
    enabled_perpetual_stats_exchanges,
    perpetual_stats,
)
from quant_tick.management.base import BaseTradeDataWithRetryCommand
from quant_tick.models import Symbol


class Command(BaseTradeDataWithRetryCommand):
    help = "Get native-cadence perpetual statistics from exchange APIs."

    def get_queryset(self):
        return Symbol.objects.filter(
            exchange__in=enabled_perpetual_stats_exchanges(),
            symbol_type=SymbolType.PERPETUAL,
        )

    def handle(self, *args, **options) -> None:
        for kwargs in super().handle(*args, **options):
            perpetual_stats(**kwargs)
