from quant_tick.constants import SymbolType
from quant_tick.exchanges.derivative_market import (
    DERIVATIVE_MARKET_SUPPORTED_EXCHANGES,
    derivative_market_data,
)
from quant_tick.management.base import BaseTradeDataWithRetryCommand
from quant_tick.models import Symbol


class Command(BaseTradeDataWithRetryCommand):
    help = "Get native-cadence perpetual market data from exchange APIs."

    def get_queryset(self):
        return Symbol.objects.filter(
            exchange__in=DERIVATIVE_MARKET_SUPPORTED_EXCHANGES,
            symbol_type=SymbolType.PERPETUAL,
        )

    def handle(self, *args, **options) -> None:
        for kwargs in super().handle(*args, **options):
            derivative_market_data(**kwargs)
