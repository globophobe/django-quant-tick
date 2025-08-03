import pandas as pd
from django.test import TestCase

from quant_tick.constants import Direction
from quant_tick.models import Candle, CandleData, Position
from quant_tick.models.strategy_types import MACrossoverStrategy

from ...base import BaseSymbolTest


class MACrossoverStrategyTest(BaseSymbolTest, TestCase):
    def setUp(self):
        super().setUp()
        self.symbol = self.get_symbol()
        self.candle = Candle.objects.create()
        self.candle.symbols.add(self.symbol)
        self.strategy = MACrossoverStrategy.objects.create(
            candle=self.candle,
            json_data={
                "moving_average_type": "sma",
                "fast_window": 2,
                "slow_window": 3,
            },
        )

    def test_backtest(self):
        """Test backtest."""
        for i in range(4):
            CandleData.objects.create(
                candle=self.candle,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}min"),
                json_data={"close": 100 + i},
            )

        self.strategy.backtest()
        positions = Position.objects.filter(strategy=self.strategy)
        self.assertEqual(len(positions), 1)
        position = positions[0]
        self.assertEqual(position.json_data["direction"], Direction.LONG.value)

    def test_backtest_with_direction_changes(self):
        """Test backtest with direction changes."""
        prices = [100, 101, 102, 101, 100, 99, 98]
        for i, price in enumerate(prices):
            CandleData.objects.create(
                candle=self.candle,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}min"),
                json_data={"close": price},
            )

        self.strategy.backtest()
        positions = Position.objects.filter(strategy=self.strategy)
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions[0].json_data["direction"], Direction.LONG.value)
        self.assertEqual(positions[1].json_data["direction"], Direction.SHORT.value)

    def test_backtest_with_existing_position(self):
        """Test backtest with existing position."""
        for i in range(4):
            candle_data = CandleData.objects.create(
                candle=self.candle,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}min"),
                json_data={"close": 100 + i},
            )

        self.strategy.backtest()

        for i in range(3):
            CandleData.objects.create(
                candle=self.candle,
                timestamp=candle_data.timestamp + pd.Timedelta(f"{i}min"),
                json_data={"close": 100 - i},
            )

        self.strategy.backtest()
        positions = Position.objects.filter(strategy=self.strategy)
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions[0].json_data["direction"], Direction.LONG.value)
        self.assertEqual(positions[1].json_data["direction"], Direction.SHORT.value)

    def test_live_trade(self):
        """Test live trade."""
        for i in range(3):
            candle_data = CandleData.objects.create(
                candle=self.candle,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}min"),
                json_data={"close": 100 + i},
            )
        self.strategy.live_trade(candle_data)
        positions = Position.objects.filter(strategy=self.strategy)
        self.assertEqual(len(positions), 1)
        position = positions[0]
        self.assertEqual(position.json_data["direction"], Direction.LONG.value)

    def test_backtest_then_live_trade(self):
        """Test backtest, then live trade."""
        for i in range(4):
            candle_data = CandleData.objects.create(
                candle=self.candle,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}min"),
                json_data={"close": 100 + i},
            )

        self.strategy.backtest()
        live_candle_data = CandleData.objects.create(
            candle=self.candle,
            timestamp=candle_data.timestamp + pd.Timedelta("1min"),
            json_data={"close": 100},
        )
        self.strategy.live_trade(live_candle_data)
        positions = Position.objects.filter(strategy=self.strategy)
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions[0].json_data["direction"], Direction.LONG.value)
        self.assertEqual(positions[1].json_data["direction"], Direction.SHORT.value)
