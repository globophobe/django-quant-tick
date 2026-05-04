from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase, TestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import exchange_candles, exchange_candles_api
from quant_tick.exchanges.hyperliquid.candles import (
    get_hyperliquid_frequency,
    get_hyperliquid_interval,
    hyperliquid_candles,
)
from quant_tick.models import ExchangeCandleData

from ..base import BaseSymbolTest


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


class ExchangeCandleFetchTest(BaseSymbolTest, TestCase):
    def get_candle_data(self, *timestamps: datetime) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "open": Decimal("1"),
                    "high": Decimal("2"),
                    "low": Decimal("0.5"),
                    "close": Decimal("1.5"),
                    "notional": Decimal("10"),
                }
                for timestamp in timestamps
            ]
        )

    def get_hyperliquid_symbol(self):
        return self.get_symbol(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
            exchange_candle_resolution="1h",
        )

    def test_exchange_candles_skips_covered_range_without_retry(self):
        symbol = self.get_hyperliquid_symbol()
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 25, 2, tzinfo=UTC)
        data = self.get_candle_data(
            timestamp_from,
            timestamp_from + pd.Timedelta("1h"),
        )
        ExchangeCandleData.write(symbol, 60, timestamp_from, timestamp_to, data)

        with patch("quant_tick.exchanges.api.exchange_candles_api") as mocked:
            exchange_candles(symbol, timestamp_from, timestamp_to)

        mocked.assert_not_called()

    def test_exchange_candles_fetches_missing_tail_without_retry(self):
        symbol = self.get_hyperliquid_symbol()
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 25, 3, tzinfo=UTC)
        ExchangeCandleData.write(
            symbol,
            60,
            timestamp_from,
            timestamp_from + pd.Timedelta("1h"),
            self.get_candle_data(timestamp_from),
        )
        fetched = self.get_candle_data(
            timestamp_from + pd.Timedelta("1h"),
            timestamp_from + pd.Timedelta("2h"),
        )

        with patch(
            "quant_tick.exchanges.api.exchange_candles_api",
            return_value=fetched,
        ) as mocked:
            exchange_candles(symbol, timestamp_from, timestamp_to)

        mocked.assert_called_once_with(
            symbol,
            timestamp_from + pd.Timedelta("1h"),
            timestamp_to,
            resolution="1h",
        )
        timestamps = list(
            ExchangeCandleData.objects.filter(symbol=symbol).values_list(
                "timestamp",
                flat=True,
            )
        )
        self.assertEqual(
            timestamps,
            [
                timestamp_from,
                timestamp_from + pd.Timedelta("1h"),
                timestamp_from + pd.Timedelta("2h"),
            ],
        )

    def test_exchange_candles_fetches_missing_middle_window_without_retry(self):
        symbol = self.get_hyperliquid_symbol()
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 25, 4, tzinfo=UTC)
        ExchangeCandleData.write(
            symbol,
            60,
            timestamp_from,
            timestamp_to,
            self.get_candle_data(
                timestamp_from,
                timestamp_from + pd.Timedelta("3h"),
            ),
        )
        fetched = self.get_candle_data(
            timestamp_from + pd.Timedelta("1h"),
            timestamp_from + pd.Timedelta("2h"),
        )

        with patch(
            "quant_tick.exchanges.api.exchange_candles_api",
            return_value=fetched,
        ) as mocked:
            exchange_candles(symbol, timestamp_from, timestamp_to)

        mocked.assert_called_once_with(
            symbol,
            timestamp_from + pd.Timedelta("1h"),
            timestamp_from + pd.Timedelta("3h"),
            resolution="1h",
        )
        timestamps = list(
            ExchangeCandleData.objects.filter(symbol=symbol).values_list(
                "timestamp",
                flat=True,
            )
        )
        self.assertEqual(
            timestamps,
            [
                timestamp_from,
                timestamp_from + pd.Timedelta("1h"),
                timestamp_from + pd.Timedelta("2h"),
                timestamp_from + pd.Timedelta("3h"),
            ],
        )

    def test_exchange_candles_retry_replaces_full_range(self):
        symbol = self.get_hyperliquid_symbol()
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 25, 2, tzinfo=UTC)
        ExchangeCandleData.write(
            symbol,
            60,
            timestamp_from,
            timestamp_to,
            self.get_candle_data(
                timestamp_from,
                timestamp_from + pd.Timedelta("1h"),
            ),
        )
        fetched = self.get_candle_data(timestamp_from + pd.Timedelta("1h"))

        with patch(
            "quant_tick.exchanges.api.exchange_candles_api",
            return_value=fetched,
        ) as mocked:
            exchange_candles(symbol, timestamp_from, timestamp_to, retry=True)

        mocked.assert_called_once_with(
            symbol,
            timestamp_from,
            timestamp_to,
            resolution="1h",
        )
        timestamps = list(
            ExchangeCandleData.objects.filter(symbol=symbol).values_list(
                "timestamp",
                flat=True,
            )
        )
        self.assertEqual(timestamps, [timestamp_from + pd.Timedelta("1h")])
