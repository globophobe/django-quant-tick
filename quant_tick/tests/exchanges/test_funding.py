from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from django.test import SimpleTestCase, TestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import FUNDING_FETCH_WINDOW, funding, funding_api
from quant_tick.exchanges.binance.funding import binance_funding
from quant_tick.exchanges.bitmex.funding import bitmex_funding
from quant_tick.exchanges.coinbase_advanced.funding import (
    coinbase_advanced_funding,
    get_coinbase_advanced_funding_url,
)
from quant_tick.exchanges.hyperliquid.api import post_hyperliquid_info
from quant_tick.exchanges.hyperliquid.funding import hyperliquid_funding
from quant_tick.models import FundingData

from ..base import BaseSymbolTest


class FundingAdapterTest(SimpleTestCase):
    def test_binance_funding_normalizes_rows(self):
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 25, 16, tzinfo=UTC)
        data = [
            {
                "fundingTime": int(
                    (timestamp_from + timedelta(milliseconds=5)).timestamp() * 1000
                ),
                "fundingRate": "0.0001",
                "markPrice": "95000.5",
            },
            {
                "fundingTime": int(
                    (timestamp_from + timedelta(hours=8)).timestamp() * 1000
                ),
                "fundingRate": "0.0002",
                "markPrice": "",
            },
        ]

        with patch(
            "quant_tick.exchanges.binance.funding.get_binance_funding_response",
            return_value=data,
        ):
            df = binance_funding("BTCUSDT", timestamp_from, timestamp_to)

        self.assertEqual(
            list(df.index),
            [
                pd.Timestamp(timestamp_from),
                pd.Timestamp(timestamp_from + timedelta(hours=8)),
            ],
        )
        self.assertEqual(df.iloc[0].funding_rate, Decimal("0.0001"))
        self.assertEqual(df.iloc[0].mark_price, Decimal("95000.5"))
        self.assertEqual(
            df.iloc[0].raw_timestamp,
            pd.Timestamp(timestamp_from + timedelta(milliseconds=5)),
        )
        self.assertEqual(df.iloc[0].timestamp_offset_ms, 5)
        self.assertFalse(df.iloc[0].timestamp_anomaly)
        self.assertEqual(df.iloc[1].funding_rate, Decimal("0.0002"))
        self.assertIsNone(df.iloc[1].mark_price)

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
                "timestamp": "2026-04-25T12:00:01.000Z",
                "fundingRate": "0.0002",
                "fundingRateDaily": "0.0006",
            }
        ]

        with patch(
            "quant_tick.exchanges.bitmex.funding.get_bitmex_funding_response",
            return_value=data,
        ) as mocked:
            df = bitmex_funding("XBTUSD", timestamp_from, timestamp_to)

        self.assertEqual(
            mocked.call_args.args[0],
            "https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=500"
            "&reverse=true&endTime=2026-04-25T16:00:00Z",
        )
        self.assertEqual(list(df.index), [pd.Timestamp("2026-04-25T12:00:00Z")])
        self.assertEqual(df.iloc[0].funding_rate, Decimal("0.0002"))
        self.assertEqual(df.iloc[0].funding_rate_daily, Decimal("0.0006"))
        self.assertEqual(
            df.iloc[0].raw_timestamp,
            pd.Timestamp("2026-04-25T12:00:01Z"),
        )
        self.assertEqual(df.iloc[0].timestamp_offset_ms, 1000)
        self.assertFalse(df.iloc[0].timestamp_anomaly)

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
                "time": int(
                    (timestamp_from + timedelta(milliseconds=48)).timestamp() * 1000
                ),
                "fundingRate": "0.0003",
                "premium": "0.0001",
            }
        ]

        with patch(
            "quant_tick.exchanges.hyperliquid.funding.get_hyperliquid_funding_response",
            return_value=data,
        ):
            df = hyperliquid_funding("BTC", timestamp_from, timestamp_to)

        self.assertEqual(list(df.index), [pd.Timestamp(timestamp_from)])
        self.assertEqual(df.iloc[0].funding_rate, Decimal("0.0003"))
        self.assertEqual(df.iloc[0].premium, Decimal("0.0001"))
        self.assertEqual(
            df.iloc[0].raw_timestamp,
            pd.Timestamp(timestamp_from + timedelta(milliseconds=48)),
        )
        self.assertEqual(df.iloc[0].timestamp_offset_ms, 48)
        self.assertFalse(df.iloc[0].timestamp_anomaly)

    def test_hyperliquid_info_uses_elapsed_time_throttle(self):
        response = SimpleNamespace(
            status_code=200,
            text="[]",
            raise_for_status=Mock(),
        )

        with (
            patch(
                "quant_tick.exchanges.hyperliquid.api.httpx.post",
                return_value=response,
            ),
            patch("quant_tick.exchanges.hyperliquid.api.time.time", side_effect=[0, 0]),
            patch("quant_tick.exchanges.hyperliquid.api.time.sleep") as mocked_sleep,
        ):
            result = post_hyperliquid_info({"type": "test"})

        self.assertEqual(result, [])
        mocked_sleep.assert_called_once_with(0.1)

    def test_coinbase_advanced_funding_url_strips_intx_suffix(self):
        url = get_coinbase_advanced_funding_url("BTC-PERP-INTX", 0)

        self.assertEqual(
            url,
            "https://api.international.coinbase.com/api/v1/instruments/"
            "BTC-PERP/funding?result_limit=100&result_offset=0",
        )

    def test_coinbase_advanced_funding_normalizes_rows(self):
        timestamp_from = datetime(2023, 3, 28, 21, tzinfo=UTC)
        timestamp_to = datetime(2026, 5, 4, 2, tzinfo=UTC)
        data = [
            {
                "event_time": "2023-03-28T21:20:25.348Z",
                "funding_rate": "0.000002",
                "mark_price": "27000.2",
            },
            {
                "event_time": "2023-05-23T10:57:49.383Z",
                "funding_rate": "0.000001",
                "mark_price": "26800.1",
            },
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
            [
                pd.Timestamp(datetime(2023, 3, 28, 21, tzinfo=UTC)),
                pd.Timestamp(datetime(2023, 5, 23, 11, tzinfo=UTC)),
                pd.Timestamp(datetime(2026, 5, 4, 1, tzinfo=UTC)),
            ],
        )
        self.assertEqual(df.iloc[0].funding_rate, Decimal("0.000002"))
        self.assertEqual(df.iloc[0].mark_price, Decimal("27000.2"))
        self.assertTrue(df.iloc[0].timestamp_anomaly)
        self.assertEqual(df.iloc[1].funding_rate, Decimal("0.000001"))
        self.assertEqual(df.iloc[1].mark_price, Decimal("26800.1"))
        self.assertFalse(df.iloc[1].timestamp_anomaly)
        self.assertEqual(df.iloc[2].funding_rate, Decimal("-0.000007"))
        self.assertEqual(df.iloc[2].mark_price, Decimal("80309"))

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


class FundingFetchTest(BaseSymbolTest, TestCase):
    def get_perpetual_symbol(self):
        return self.get_symbol(
            exchange=Exchange.BITMEX,
            api_symbol="XBTUSD",
            symbol_type=SymbolType.PERPETUAL,
        )

    def get_coinbase_advanced_symbol(self):
        return self.get_symbol(
            exchange=Exchange.COINBASE_ADVANCED,
            api_symbol="BTC-PERP-INTX",
            symbol_type=SymbolType.PERPETUAL,
        )

    def write_existing_rows(self, symbol, timestamp_from):
        FundingData.objects.bulk_create(
            [
                FundingData(
                    symbol=symbol,
                    timestamp=timestamp_from,
                    funding_rate=Decimal("0.0001"),
                ),
                FundingData(
                    symbol=symbol,
                    timestamp=timestamp_from + timedelta(hours=8),
                    funding_rate=Decimal("0.0002"),
                ),
            ]
        )

    def test_funding_skips_fetch_when_latest_row_covers_requested_window(self):
        symbol = self.get_perpetual_symbol()
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 25, 12, tzinfo=UTC)
        self.write_existing_rows(symbol, timestamp_from)

        with patch("quant_tick.exchanges.api.funding_api") as mocked:
            funding(symbol, timestamp_from, timestamp_to)

        mocked.assert_not_called()
        self.assertEqual(FundingData.objects.filter(symbol=symbol).count(), 2)

    def test_funding_fetches_only_after_latest_row_when_next_event_is_due(self):
        symbol = self.get_perpetual_symbol()
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 25, 17, tzinfo=UTC)
        self.write_existing_rows(symbol, timestamp_from)
        data = pd.DataFrame(
            [
                {
                    "timestamp": timestamp_from + timedelta(hours=16),
                    "funding_rate": Decimal("0.0003"),
                }
            ]
        )

        with patch(
            "quant_tick.exchanges.api.funding_api",
            return_value=data,
        ) as mocked:
            funding(symbol, timestamp_from, timestamp_to)

        mocked.assert_called_once_with(
            symbol,
            timestamp_from + timedelta(hours=8, microseconds=1),
            timestamp_to,
        )
        rows = list(FundingData.objects.filter(symbol=symbol))
        self.assertEqual(
            [row.timestamp for row in rows],
            [
                timestamp_from,
                timestamp_from + timedelta(hours=8),
                timestamp_from + timedelta(hours=16),
            ],
        )

    def test_funding_empty_history_backfills_newest_window_first(self):
        symbol = self.get_perpetual_symbol()
        timestamp_from = datetime(2025, 1, 1, tzinfo=UTC)
        timestamp_to = datetime(2026, 2, 1, tzinfo=UTC)

        def fetch(_symbol, ts_from, ts_to):
            return pd.DataFrame(
                [
                    {
                        "timestamp": ts_from,
                        "funding_rate": Decimal("0.0001"),
                    }
                ]
            )

        with patch(
            "quant_tick.exchanges.api.funding_api",
            side_effect=fetch,
        ) as mocked:
            funding(symbol, timestamp_from, timestamp_to)

        first_call = mocked.call_args_list[0].args
        self.assertEqual(
            first_call,
            (
                symbol,
                timestamp_to - FUNDING_FETCH_WINDOW,
                timestamp_to,
            ),
        )

    def test_coinbase_advanced_funding_uses_single_offset_pagination_pass(self):
        symbol = self.get_coinbase_advanced_symbol()
        timestamp_from = datetime(2025, 1, 1, tzinfo=UTC)
        timestamp_to = datetime(2026, 2, 1, tzinfo=UTC)
        data = pd.DataFrame(
            [
                {
                    "timestamp": timestamp_to - timedelta(hours=1),
                    "funding_rate": Decimal("0.0001"),
                }
            ]
        )

        with patch(
            "quant_tick.exchanges.api.funding_api",
            return_value=data,
        ) as mocked:
            funding(symbol, timestamp_from, timestamp_to)

        mocked.assert_called_once_with(symbol, timestamp_from, timestamp_to)

    def test_retry_refetches_requested_window(self):
        symbol = self.get_perpetual_symbol()
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 25, 16, tzinfo=UTC)
        self.write_existing_rows(symbol, timestamp_from)
        data = pd.DataFrame(
            [
                {
                    "timestamp": timestamp_from + timedelta(hours=8),
                    "funding_rate": Decimal("0.0004"),
                }
            ]
        )

        with patch(
            "quant_tick.exchanges.api.funding_api",
            return_value=data,
        ) as mocked:
            funding(symbol, timestamp_from, timestamp_to, retry=True)

        mocked.assert_called_once_with(symbol, timestamp_from, timestamp_to)
        rows = list(FundingData.objects.filter(symbol=symbol))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].timestamp, timestamp_from + timedelta(hours=8))
        self.assertEqual(rows[0].funding_rate, Decimal("0.0004"))
