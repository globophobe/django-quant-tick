from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import exchange_candles_api
from quant_tick.exchanges.hyperliquid.candles import (
    get_hyperliquid_frequency,
    get_hyperliquid_interval,
    hyperliquid_candles,
)


class HyperliquidCandleTest(SimpleTestCase):
    def test_get_hyperliquid_interval_supports_8h(self):
        self.assertEqual(get_hyperliquid_interval("8h"), "8h")
        self.assertEqual(get_hyperliquid_frequency("8h"), 480)

    def test_hyperliquid_candles_normalizes_base_volume_as_notional(self):
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 25, 1, tzinfo=UTC)
        data = [
            {
                "t": int(timestamp_from.timestamp() * 1000),
                "o": "1",
                "h": "2",
                "l": "0.5",
                "c": "1.5",
                "v": "10",
                "n": 3,
            }
        ]

        with patch(
            "quant_tick.exchanges.hyperliquid.candles.get_hyperliquid_candle_response",
            return_value=data,
        ):
            df = hyperliquid_candles("BTC", timestamp_from, timestamp_to)

        self.assertEqual(list(df.index), [pd.Timestamp(timestamp_from)])
        self.assertEqual(df.iloc[0].open, Decimal("1"))
        self.assertEqual(df.iloc[0].notional, Decimal("10"))
        self.assertEqual(df.iloc[0].trades, 3)

    def test_exchange_candles_api_dispatches_hyperliquid(self):
        symbol = SimpleNamespace(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
            exchange_candle_resolution="",
            clamp_timestamp_range=lambda ts_from, ts_to: (ts_from, ts_to),
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 26, tzinfo=UTC)
        expected = pd.DataFrame([])

        with patch(
            "quant_tick.exchanges.api.hyperliquid_candles",
            return_value=expected,
        ) as mocked:
            result = exchange_candles_api(
                symbol,
                timestamp_from,
                timestamp_to,
                resolution="8h",
            )

        mocked.assert_called_once_with(
            "BTC",
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            resolution="8h",
        )
        self.assertTrue(result.equals(expected))

    def test_exchange_candles_api_uses_symbol_resolution(self):
        symbol = SimpleNamespace(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
            exchange_candle_resolution="8h",
            clamp_timestamp_range=lambda ts_from, ts_to: (ts_from, ts_to),
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 26, tzinfo=UTC)
        expected = pd.DataFrame([])

        with patch(
            "quant_tick.exchanges.api.hyperliquid_candles",
            return_value=expected,
        ) as mocked:
            result = exchange_candles_api(symbol, timestamp_from, timestamp_to)

        mocked.assert_called_once_with(
            "BTC",
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            resolution="8h",
        )
        self.assertTrue(result.equals(expected))

    def test_exchange_candles_api_requires_resolution(self):
        symbol = SimpleNamespace(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
            exchange_candle_resolution="",
            clamp_timestamp_range=lambda ts_from, ts_to: (ts_from, ts_to),
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 26, tzinfo=UTC)

        with self.assertRaises(ValueError):
            exchange_candles_api(symbol, timestamp_from, timestamp_to)
