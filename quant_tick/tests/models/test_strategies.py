from django.contrib.contenttypes.models import ContentType
from django.test import TestCase

from quant_tick.models import Candle, MACrossoverStrategy

from ..base import BaseSymbolTest


class BaseStrategyTest(BaseSymbolTest, TestCase):
    """Base strategy test."""

    def setUp(self):
        """Set up."""
        ContentType.objects.clear_cache()
        super().setUp()
        self.symbol = self.get_symbol()


class WarmupParityTest(BaseSymbolTest, TestCase):
    """Test that training and inference have matching warmup behavior."""

    def setUp(self) -> None:
        super().setUp()
        self.symbol = self.get_symbol()
        self.candle = Candle.objects.create(symbol=self.symbol)
        self.strategy = MACrossoverStrategy.objects.create(
            candle=self.candle,
            json_data={
                "moving_average_type": "sma",
                "fast_window": 2,
                "slow_window": 3,
            },
        )
