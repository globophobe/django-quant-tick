from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import funding_api
from quant_tick.exchanges.binance.funding import binance_funding
from quant_tick.exchanges.bitmex.funding import bitmex_funding
from quant_tick.exchanges.coinbase_advanced.funding import (
    coinbase_advanced_funding,
    get_coinbase_advanced_funding_url,
)
from quant_tick.exchanges.hyperliquid.funding import hyperliquid_funding


class FundingAdapterTest(SimpleTestCase):
    def test_binance_funding_normalizes_rows(self):
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 25, 16, tzinfo=UTC)
        data = [
            {
                "fundingTime": int(timestamp_from.timestamp() * 1000),
                "fundingRate": "0.0001",
                "markPrice": "95000.5",
            }
        ]

        with patch(
            "quant_tick.exchanges.binance.funding.get_binance_funding_response",
            return_value=data,
        ):
            df = binance_funding("BTCUSDT", timestamp_from, timestamp_to)

        self.assertEqual(list(df.index), [pd.Timestamp(timestamp_from)])
        self.assertEqual(df.iloc[0].funding_rate, Decimal("0.0001"))
        self.assertEqual(df.iloc[0].mark_price, Decimal("95000.5"))

    def test_binance_funding_response_uses_shared_api_helper(self):
        from quant_tick.exchanges.binance.funding import get_binance_funding_response

        with patch(
            "quant_tick.exchanges.binance.funding.get_binance_api_response",
            return_value=[],
        ) as mocked:
            result = get_binance_funding_response("https://example.test/funding")

        self.assertEqual(result, [])
        self.assertEqual(mocked.call_args.args[1], "https://example.test/funding")
        self.assertFalse(mocked.call_args.kwargs["reverse"])

    def test_bitmex_funding_preserves_extra_rates(self):
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 25, 16, tzinfo=UTC)
        data = [
            {
                "timestamp": "2026-04-25T08:00:00.000Z",
                "fundingRate": "0.0002",
                "fundingRateDaily": "0.0006",
            }
        ]

        with patch(
            "quant_tick.exchanges.bitmex.funding.get_bitmex_funding_response",
            return_value=data,
        ):
            df = bitmex_funding("XBTUSD", timestamp_from, timestamp_to)

        self.assertEqual(df.iloc[0].funding_rate, Decimal("0.0002"))
        self.assertEqual(df.iloc[0].funding_rate_daily, Decimal("0.0006"))

    def test_bitmex_funding_response_uses_shared_api_helper(self):
        from quant_tick.exchanges.bitmex.funding import get_bitmex_funding_response

        with patch(
            "quant_tick.exchanges.bitmex.funding.get_bitmex_api_response",
            return_value=[],
        ) as mocked:
            result = get_bitmex_funding_response("https://example.test/funding")

        self.assertEqual(result, [])
        self.assertEqual(mocked.call_args.args[1], "https://example.test/funding")

    def test_hyperliquid_funding_normalizes_rows(self):
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 25, 2, tzinfo=UTC)
        data = [
            {
                "time": int(timestamp_from.timestamp() * 1000),
                "fundingRate": "0.0003",
                "premium": "0.0001",
            }
        ]

        with patch(
            "quant_tick.exchanges.hyperliquid.funding.get_hyperliquid_funding_response",
            return_value=data,
        ):
            df = hyperliquid_funding("BTC", timestamp_from, timestamp_to)

        self.assertEqual(df.iloc[0].funding_rate, Decimal("0.0003"))
        self.assertEqual(df.iloc[0].premium, Decimal("0.0001"))

    def test_coinbase_advanced_funding_url_strips_intx_suffix(self):
        url = get_coinbase_advanced_funding_url("BTC-PERP-INTX", 0)

        self.assertEqual(
            url,
            "https://api.international.coinbase.com/api/v1/instruments/"
            "BTC-PERP/funding?result_limit=100&result_offset=0",
        )

    def test_coinbase_advanced_funding_normalizes_rows(self):
        timestamp_from = datetime(2026, 5, 4, tzinfo=UTC)
        timestamp_to = datetime(2026, 5, 4, 2, tzinfo=UTC)
        data = [
            {
                "event_time": "2026-05-04T01:00:00Z",
                "funding_rate": "-0.000007",
                "mark_price": "80309",
            }
        ]

        with patch(
            "quant_tick.exchanges.coinbase_advanced.funding."
            "get_coinbase_advanced_funding_response",
            return_value=data,
        ):
            df = coinbase_advanced_funding(
                "BTC-PERP-INTX",
                timestamp_from,
                timestamp_to,
            )

        self.assertEqual(
            list(df.index),
            [pd.Timestamp(datetime(2026, 5, 4, 1, tzinfo=UTC))],
        )
        self.assertEqual(df.iloc[0].funding_rate, Decimal("-0.000007"))
        self.assertEqual(df.iloc[0].mark_price, Decimal("80309"))

    def test_funding_api_dispatches_hyperliquid_perpetuals(self):
        symbol = SimpleNamespace(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
            clamp_timestamp_range=lambda ts_from, ts_to: (ts_from, ts_to),
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 26, tzinfo=UTC)
        expected = pd.DataFrame([])

        with patch(
            "quant_tick.exchanges.api.hyperliquid_funding",
            return_value=expected,
        ) as mocked:
            result = funding_api(symbol, timestamp_from, timestamp_to)

        mocked.assert_called_once_with("BTC", timestamp_from, timestamp_to)
        self.assertTrue(result.equals(expected))

    def test_funding_api_dispatches_coinbase_advanced_perpetuals(self):
        symbol = SimpleNamespace(
            exchange=Exchange.COINBASE_ADVANCED,
            api_symbol="BTC-PERP-INTX",
            symbol_type=SymbolType.PERPETUAL,
            clamp_timestamp_range=lambda ts_from, ts_to: (ts_from, ts_to),
        )
        timestamp_from = datetime(2026, 5, 4, tzinfo=UTC)
        timestamp_to = datetime(2026, 5, 5, tzinfo=UTC)
        expected = pd.DataFrame([])

        with patch(
            "quant_tick.exchanges.api.coinbase_advanced_funding",
            return_value=expected,
        ) as mocked:
            result = funding_api(symbol, timestamp_from, timestamp_to)

        mocked.assert_called_once_with("BTC-PERP-INTX", timestamp_from, timestamp_to)
        self.assertTrue(result.equals(expected))

    def test_funding_api_dispatches_binance_futures_perpetuals(self):
        symbol = SimpleNamespace(
            exchange=Exchange.BINANCE_FUTURES,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
            clamp_timestamp_range=lambda ts_from, ts_to: (ts_from, ts_to),
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 26, tzinfo=UTC)
        expected = pd.DataFrame([])

        with patch(
            "quant_tick.exchanges.api.binance_funding",
            return_value=expected,
        ) as mocked:
            result = funding_api(symbol, timestamp_from, timestamp_to)

        mocked.assert_called_once_with("BTCUSDT", timestamp_from, timestamp_to)
        self.assertTrue(result.equals(expected))

    def test_funding_api_rejects_plain_binance_perpetuals(self):
        symbol = SimpleNamespace(
            exchange=Exchange.BINANCE,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
            clamp_timestamp_range=lambda ts_from, ts_to: (ts_from, ts_to),
        )

        with self.assertRaises(ValueError):
            funding_api(
                symbol,
                datetime(2026, 4, 25, tzinfo=UTC),
                datetime(2026, 4, 26, tzinfo=UTC),
            )

    def test_funding_api_rejects_spot_symbols(self):
        symbol = SimpleNamespace(
            exchange=Exchange.BINANCE,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.SPOT,
        )

        with self.assertRaises(ValueError):
            funding_api(
                symbol,
                datetime(2026, 4, 25, tzinfo=UTC),
                datetime(2026, 4, 26, tzinfo=UTC),
            )
