import logging

from django.db.models import QuerySet

from quant_candles.controllers import aggregate_candles
from quant_candles.management.base import BaseCandleCommand
from quant_candles.models import Candle

logger = logging.getLogger(__name__)


class Command(BaseCandleCommand):
    help = "Create candles from trade data."

    def get_queryset(self) -> QuerySet:
        """Get queryset."""
        return Candle.objects.filter(is_active=True).prefetch_related("symbols")

    def handle(self, *args, **options) -> None:
        """Run command."""
        kwargs = super().handle(*args, **options)
        aggregate_candles(**kwargs)
