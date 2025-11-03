import numpy as np
import pandas as pd
from django.test import TestCase

from quant_tick.management.commands.ml_backtest import simulate_trades
from quant_tick.models import Candle, GlobalSymbol, MLConfig, MLRun, Position, Symbol


class MLBacktestContributionsTestCase(TestCase):
    """Test feature contributions logging in ml_backtest."""

    def setUp(self):
        """Create test fixtures."""
        self.global_symbol = GlobalSymbol.objects.create(name="BTC-USD")
        self.symbol = Symbol.objects.create(
            global_symbol=self.global_symbol,
            exchange="test_exchange",
            api_symbol="BTC-USD",
        )

        self.candle = Candle.objects.create(code_name="btc-1h")
        self.candle.symbols.add(self.symbol)

        self.ml_config = MLConfig.objects.create(
            code_name="test-backtest-contrib",
            candle=self.candle,
            symbol=self.symbol,
            json_data={
                "timestamp_from": "2025-01-01",
                "timestamp_to": "2025-01-10",
                "pt_mult": 2.0,
                "sl_mult": 1.0,
                "max_holding_bars": 48,
            },
            is_active=True,
        )

        self.ml_run = MLRun.objects.create(
            ml_config=self.ml_config,
            timestamp_from=pd.Timestamp("2025-01-01"),
            timestamp_to=pd.Timestamp("2025-01-10"),
            metrics={},
            status="running",
        )

    def test_simulate_trades_without_contributions(self):
        """Test simulate_trades without feature contributions (default behavior)."""
        n_bars = 30
        prices = np.array([100.0] * 10 + [102.0] * 10 + [98.0] * 10)
        timestamps = pd.date_range("2025-01-01", periods=n_bars, freq="1h")
        highs = prices * 1.01
        lows = prices * 0.99
        volatilities = np.full(n_bars, 0.01)

        signals = [0] * n_bars
        signals[5] = 1
        signals[15] = -1

        trades, equity = simulate_trades(
            signals,
            prices,
            timestamps,
            self.ml_run,
            costs_bps=5,
            pt_mult=2.0,
            sl_mult=1.0,
            max_holding_bars=48,
            highs=highs,
            lows=lows,
            volatilities=volatilities,
        )

        self.assertEqual(len(trades), 2)
        positions = Position.objects.filter(ml_run=self.ml_run)
        self.assertEqual(positions.count(), 2)

        for position in positions:
            self.assertIsNone(position.json_data)

    def test_simulate_trades_with_contributions(self):
        """Test simulate_trades with feature contributions enabled."""
        n_bars = 30
        prices = np.array([100.0] * 10 + [102.0] * 10 + [98.0] * 10)
        timestamps = pd.date_range("2025-01-01", periods=n_bars, freq="1h")
        highs = prices * 1.01
        lows = prices * 0.99
        volatilities = np.full(n_bars, 0.01)

        signals = [0] * n_bars
        signals[5] = 1
        signals[15] = -1

        feature_contributions = [
            {"feature_0": 0.5, "feature_1": -0.3, "feature_2": 0.1},
            {"feature_0": 0.2, "feature_1": -0.5, "feature_2": 0.3},
        ]

        trades, equity = simulate_trades(
            signals,
            prices,
            timestamps,
            self.ml_run,
            costs_bps=5,
            pt_mult=2.0,
            sl_mult=1.0,
            max_holding_bars=48,
            highs=highs,
            lows=lows,
            volatilities=volatilities,
            feature_contributions=feature_contributions,
        )

        self.assertEqual(len(trades), 2)
        positions = Position.objects.filter(ml_run=self.ml_run).order_by("entry_timestamp")
        self.assertEqual(positions.count(), 2)

        for i, position in enumerate(positions):
            self.assertIsNotNone(position.json_data)
            self.assertIn("feature_contributions", position.json_data)

            contrib = position.json_data["feature_contributions"]
            self.assertEqual(len(contrib), 3)
            self.assertIn("feature_0", contrib)
            self.assertIn("feature_1", contrib)
            self.assertIn("feature_2", contrib)

            expected = feature_contributions[i]
            self.assertEqual(contrib["feature_0"], expected["feature_0"])
            self.assertEqual(contrib["feature_1"], expected["feature_1"])
            self.assertEqual(contrib["feature_2"], expected["feature_2"])

    def test_contributions_alignment_with_signals(self):
        """Test that contributions align correctly with non-zero signals."""
        n_bars = 50
        prices = np.full(n_bars, 100.0)
        timestamps = pd.date_range("2025-01-01", periods=n_bars, freq="1h")
        volatilities = np.full(n_bars, 0.01)

        signals = [0] * n_bars
        signals[10] = 1
        signals[20] = 1
        signals[30] = -1

        feature_contributions = [
            {"f1": 0.1, "f2": 0.2},
            {"f1": 0.3, "f2": 0.4},
            {"f1": 0.5, "f2": 0.6},
        ]

        trades, equity = simulate_trades(
            signals,
            prices,
            timestamps,
            self.ml_run,
            volatilities=volatilities,
            feature_contributions=feature_contributions,
        )

        self.assertEqual(len(trades), 3)
        positions = Position.objects.filter(ml_run=self.ml_run).order_by("entry_timestamp")
        self.assertEqual(positions.count(), 3)

        for i, position in enumerate(positions):
            contrib = position.json_data["feature_contributions"]
            expected = feature_contributions[i]
            self.assertEqual(contrib, expected)
