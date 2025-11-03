from decimal import Decimal

import numpy as np
import pandas as pd
from django.test import TestCase

from quant_tick.constants import (
    Exchange,
    ExitReason,
    PositionStatus,
    PositionType,
    SymbolType,
)
from quant_tick.management.commands.ml_backtest import simulate_trades
from quant_tick.models import Candle, GlobalSymbol, MLConfig, MLRun, Position, Symbol


class SimulateTradesTestCase(TestCase):
    """Test simulate_trades() basic functionality."""

    def setUp(self):
        """Create test fixtures."""
        self.global_symbol = GlobalSymbol.objects.create(name="BTC")
        self.symbol = Symbol.objects.create(
            code_name="btc-test",
            global_symbol=self.global_symbol,
            exchange=Exchange.COINBASE,
            symbol_type=SymbolType.SPOT,
            api_symbol="BTC-USD"
        )
        self.candle = Candle.objects.create()
        self.ml_config = MLConfig.objects.create(
            code_name="test-config",
            candle=self.candle,
            symbol=self.symbol,
            json_data={
                "timestamp_from": "2025-01-01",
                "timestamp_to": "2025-01-10"
            }
        )

    def test_simulate_trades_basic(self):
        """Test simulate_trades with basic long/short signals."""
        n_bars = 30
        prices = np.array([100.0] * 10 + [102.0] * 10 + [98.0] * 10)
        timestamps = pd.date_range("2025-01-01", periods=n_bars, freq="1h")
        highs = prices * 1.01
        lows = prices * 0.99
        volatilities = np.full(n_bars, 0.01)

        signals = [0] * n_bars
        signals[5] = 1  # Long signal
        signals[15] = -1  # Short signal

        ml_run = MLRun.objects.create(
            ml_config=self.ml_config,
            timestamp_from=pd.Timestamp("2025-01-01").to_pydatetime(),
            timestamp_to=pd.Timestamp("2025-01-10").to_pydatetime(),
            metrics={},
            feature_importances={},
            status="running"
        )

        trades, equity = simulate_trades(
            signals, prices, timestamps, ml_run,
            costs_bps=5, pt_mult=2.0, sl_mult=1.0, max_holding_bars=20,
            highs=highs, lows=lows, volatilities=volatilities
        )

        # Verify basic structure
        self.assertEqual(len(trades), 2)
        self.assertIsInstance(equity, Decimal)

        # Verify trade fields
        for trade in trades:
            self.assertIn("entry_idx", trade)
            self.assertIn("exit_idx", trade)
            self.assertIn("direction", trade)
            self.assertIn("pnl", trade)
            self.assertIn("equity", trade)
            self.assertIn("position_id", trade)
            self.assertIsNotNone(trade["position_id"])

        # Verify Position records created
        positions = Position.objects.filter(ml_run=ml_run).order_by("entry_timestamp")
        self.assertEqual(positions.count(), 2)

        # Verify Position fields
        for position in positions:
            self.assertEqual(position.position_type, PositionType.BACKTEST)
            self.assertEqual(position.status, PositionStatus.CLOSED)
            self.assertIsNotNone(position.entry_price)
            self.assertIsNotNone(position.exit_price)
            self.assertIsNotNone(position.take_profit)
            self.assertIsNotNone(position.stop_loss)
            self.assertIn(position.exit_reason, [
                ExitReason.TAKE_PROFIT,
                ExitReason.STOP_LOSS,
                ExitReason.MAX_DURATION
            ])

        # Verify long and short sides
        long_position = positions.filter(side=1).first()
        short_position = positions.filter(side=-1).first()

        self.assertIsNotNone(long_position)
        self.assertIsNotNone(short_position)

        # Verify TP/SL logic for long
        self.assertGreater(long_position.take_profit, long_position.entry_price)
        self.assertLess(long_position.stop_loss, long_position.entry_price)

        # Verify TP/SL logic for short
        self.assertLess(short_position.take_profit, short_position.entry_price)
        self.assertGreater(short_position.stop_loss, short_position.entry_price)

    def test_simulate_trades_no_signals(self):
        """Test simulate_trades with no signals."""
        n_bars = 50
        prices = np.full(n_bars, 100.0)
        timestamps = pd.date_range("2025-01-01", periods=n_bars, freq="1h")
        signals = [0] * n_bars

        ml_run = MLRun.objects.create(
            ml_config=self.ml_config,
            timestamp_from=pd.Timestamp("2025-01-01").to_pydatetime(),
            timestamp_to=pd.Timestamp("2025-01-10").to_pydatetime(),
            metrics={},
            feature_importances={},
            status="running"
        )

        trades, equity = simulate_trades(
            signals, prices, timestamps, ml_run,
            costs_bps=5, pt_mult=2.0, sl_mult=1.0, max_holding_bars=10
        )

        self.assertEqual(len(trades), 0)
        self.assertEqual(equity, Decimal("1.0"))
        self.assertEqual(Position.objects.filter(ml_run=ml_run).count(), 0)
