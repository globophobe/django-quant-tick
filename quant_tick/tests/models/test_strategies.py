import numpy as np
import pandas as pd
from django.contrib.contenttypes.models import ContentType
from django.test import TestCase

from quant_tick.constants import Exchange
from quant_tick.models import Candle, MACrossoverStrategy, Symbol

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
        self.candle = Candle.objects.create()
        self.candle.symbols.add(self.symbol)
        self.strategy = MACrossoverStrategy.objects.create(
            candle=self.candle,
            symbol=self.symbol,
            json_data={
                "moving_average_type": "sma",
                "fast_window": 2,
                "slow_window": 3,
            },
        )

class MultiExchangeCanonicalTest(BaseSymbolTest, TestCase):
    """Test canonical exchange selection for multi-exchange candles."""

    def _make_strategy(self, exchange: Exchange) -> MACrossoverStrategy:
        symbol = Symbol.objects.create(
            global_symbol=self.global_symbol,
            exchange=exchange,
            api_symbol=f"{exchange}-btc",
        )
        candle = Candle.objects.create()
        candle.symbols.add(symbol)
        return MACrossoverStrategy.objects.create(
            candle=candle,
            symbol=symbol,
            json_data={},
        )

    @staticmethod
    def _make_multi_exchange_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min"),
                "coinbaseClose": [100 + i for i in range(10)],
                "binanceClose": [101 + i for i in range(10)],
                "coinbaseOpen": [99 + i for i in range(10)],
                "binanceOpen": [100 + i for i in range(10)],
                "coinbaseHigh": [102 + i for i in range(10)],
                "binanceHigh": [103 + i for i in range(10)],
                "coinbaseLow": [98 + i for i in range(10)],
                "binanceLow": [99 + i for i in range(10)],
                "coinbaseVolume": [1000] * 10,
                "binanceVolume": [1100] * 10,
            }
        )

    def test_canonical_uses_specified_exchange(self):
        """Canonical exchange matches the strategy symbol exchange."""
        strategy = self._make_strategy(Exchange.COINBASE)
        df = self._make_multi_exchange_frame()

        result = strategy.compute_features(df)

        pd.testing.assert_series_equal(
            result["close"], df["coinbaseClose"], check_names=False
        )
        self.assertIn("basisBinance", result.columns)
        self.assertIn("basisPctBinance", result.columns)

    def test_canonical_uses_binance_when_specified(self):
        """Canonical exchange can be binance."""
        strategy = self._make_strategy(Exchange.BINANCE)
        df = self._make_multi_exchange_frame()

        result = strategy.compute_features(df)

        pd.testing.assert_series_equal(
            result["close"], df["binanceClose"], check_names=False
        )
        self.assertIn("basisCoinbase", result.columns)
        self.assertIn("basisPctCoinbase", result.columns)

    def test_missing_canonical_raises_error(self):
        """Raises error if canonical exchange is not present."""
        strategy = self._make_strategy(Exchange.BITMEX)
        df = self._make_multi_exchange_frame()

        with self.assertRaises(ValueError) as ctx:
            strategy.compute_features(df)

        self.assertIn("bitmex", str(ctx.exception))
