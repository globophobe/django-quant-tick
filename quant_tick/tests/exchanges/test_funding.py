from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from django.test import SimpleTestCase, TestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import FUNDING_FETCH_WINDOW, funding, funding_api
from quant_tick.exchanges.binance.funding import binance_funding
from quant_tick.exchanges.bitfinex.funding import bitfinex_funding
from quant_tick.exchanges.bitmex.funding import bitmex_funding
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

    def test_bitfinex_funding_uses_latest_status_before_event(self):
        timestamp_from = datetime(2026, 5, 5, tzinfo=UTC)
        timestamp_to = datetime(2026, 5, 5, 8, tzinfo=UTC)

        def status_row(
            status_timestamp,
            event_timestamp,
            accrued,
            current_funding="0.00001",
        ):
            row = [None] * 23
            row[0] = int(status_timestamp.timestamp() * 1000)
            row[7] = int(event_timestamp.timestamp() * 1000)
            row[8] = accrued
            row[9] = 9234
            row[11] = current_funding
            row[14] = "79846.619"
            row[17] = "8687.0615269"
            row[21] = "-0.0005"
            row[22] = "0.0025"
            return row

        data = [
            status_row(
                timestamp_from + timedelta(minutes=1),
                timestamp_to,
                "0.000001",
                current_funding="0",
            ),
            status_row(
                timestamp_from - timedelta(seconds=2),
                timestamp_from,
                "0.00044342",
            ),
            status_row(
                timestamp_from - timedelta(minutes=2),
                timestamp_from,
                "0.00044063",
            ),
        ]

        with patch(
            "quant_tick.exchanges.bitfinex.funding.get_bitfinex_funding_response",
            return_value=data,
        ) as mocked:
            df = bitfinex_funding("tBTCF0:USTF0", timestamp_from, timestamp_to)

        self.assertIn(
            "/status/deriv/tBTCF0:USTF0/hist",
            mocked.call_args.args[0],
        )
        self.assertIn("limit=5000", mocked.call_args.args[0])
        self.assertEqual(list(df.index), [pd.Timestamp(timestamp_from)])
        self.assertEqual(df.iloc[0].funding_rate, Decimal("0.00044342"))
        self.assertEqual(
            df.iloc[0].status_timestamp,
            pd.Timestamp(timestamp_from - timedelta(seconds=2)),
        )
        self.assertEqual(df.iloc[0].current_funding, Decimal("0.00001"))
        self.assertEqual(df.iloc[0].mark_price, Decimal("79846.619"))
        self.assertIsNone(df.iloc[0].raw_timestamp)

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

    def test_funding_api_dispatches_bitfinex_perpetuals(self):
        symbol = SimpleNamespace(
            exchange=Exchange.BITFINEX,
            api_symbol="tBTCF0:USTF0",
            symbol_type=SymbolType.PERPETUAL,
            clamp_timestamp_range=lambda ts_from, ts_to: (ts_from, ts_to),
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 26, tzinfo=UTC)
        expected = pd.DataFrame([])

        with patch(
            "quant_tick.exchanges.api.bitfinex_funding",
            return_value=expected,
        ) as mocked:
            result = funding_api(symbol, timestamp_from, timestamp_to)

        mocked.assert_called_once_with("tBTCF0:USTF0", timestamp_from, timestamp_to)
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
    def get_binance_futures_symbol(self):
        return self.get_symbol(
            exchange=Exchange.BINANCE_FUTURES,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )

    def get_bitmex_symbol(self):
        return self.get_symbol(
            exchange=Exchange.BITMEX,
            api_symbol="XBTUSD",
            symbol_type=SymbolType.PERPETUAL,
        )

    def write_existing_rows(self, symbol, *timestamps):
        FundingData.objects.bulk_create(
            [
                FundingData(
                    symbol=symbol,
                    timestamp=timestamp,
                    funding_rate=Decimal(f"0.000{index + 1}"),
                )
                for index, timestamp in enumerate(timestamps)
            ]
        )

    def test_funding_skips_fetch_when_requested_window_is_covered(self):
        symbol = self.get_binance_futures_symbol()
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 25, 16, tzinfo=UTC)
        self.write_existing_rows(
            symbol,
            timestamp_from,
            timestamp_from + timedelta(hours=8),
        )

        with patch("quant_tick.exchanges.api.funding_api") as mocked:
            funding(symbol, timestamp_from, timestamp_to)

        mocked.assert_not_called()
        self.assertEqual(FundingData.objects.filter(symbol=symbol).count(), 2)

    def test_funding_fetches_missing_tail_without_retry(self):
        symbol = self.get_binance_futures_symbol()
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 26, tzinfo=UTC)
        self.write_existing_rows(
            symbol,
            timestamp_from,
            timestamp_from + timedelta(hours=8),
        )
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
            timestamp_from + timedelta(hours=16),
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

    def test_funding_fetches_missing_middle_window_without_retry(self):
        symbol = self.get_bitmex_symbol()
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 26, tzinfo=UTC)
        self.write_existing_rows(
            symbol,
            timestamp_from + timedelta(hours=4),
            timestamp_from + timedelta(hours=20),
        )
        data = pd.DataFrame(
            [
                {
                    "timestamp": timestamp_from + timedelta(hours=12),
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
            timestamp_from + timedelta(hours=12),
            timestamp_from + timedelta(hours=20),
        )
        rows = list(FundingData.objects.filter(symbol=symbol))
        self.assertEqual(
            [row.timestamp for row in rows],
            [
                timestamp_from + timedelta(hours=4),
                timestamp_from + timedelta(hours=12),
                timestamp_from + timedelta(hours=20),
            ],
        )

    def test_funding_empty_history_backfills_newest_window_first(self):
        symbol = self.get_binance_futures_symbol()
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

    def test_retry_refetches_requested_window(self):
        symbol = self.get_binance_futures_symbol()
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 25, 16, tzinfo=UTC)
        self.write_existing_rows(
            symbol,
            timestamp_from,
            timestamp_from + timedelta(hours=8),
        )
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
