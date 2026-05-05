from quant_tick.constants import SymbolType
from quant_tick.exchanges.api import funding
from quant_tick.management.base import BaseTradeDataWithRetryCommand
from quant_tick.models import Symbol


class Command(BaseTradeDataWithRetryCommand):
    help = "Get perpetual funding data from exchange APIs."

    def get_queryset(self):
        return Symbol.objects.filter(symbol_type=SymbolType.PERPETUAL)

    def handle(self, *args, **options) -> None:
        for k in super().handle(*args, **options):
            funding(**k)
