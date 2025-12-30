from quant_tick.exchanges.binance.funding import collect_funding_rates
from quant_tick.management.base import BaseTradeDataCommand


class Command(BaseTradeDataCommand):
    """Collect funding rates from exchange API."""

    help = "Get funding rates from exchange API."

    def handle(self, *args, **options) -> None:
        """Run command."""
        kwargs = super().handle(*args, **options)
        for k in kwargs:
            collect_funding_rates(
                symbol=k["symbol"],
                timestamp_from=k["timestamp_from"],
                timestamp_to=k["timestamp_to"],
            )
