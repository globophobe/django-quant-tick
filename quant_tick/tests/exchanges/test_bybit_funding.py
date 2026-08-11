from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import call, patch

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import funding_api
from quant_tick.exchanges.bybit.funding import (
    BybitFunding,
    _fetch_funding_rows,
    bybit_funding,
    bybit_open_interest,
    get_bybit_funding_interval,
)


def millis(timestamp: datetime) -> str:
    return str(int(timestamp.timestamp() * 1000))


class BybitFundingTest(SimpleTestCase):
    def test_funding_interval_uses_instrument_metadata(self):
        with patch(
            "quant_tick.exchanges.bybit.funding.get_bybit_result",
            return_value={
                "list": [
                    {"symbol": "BTCUSDT", "fundingInterval": "240"},
                ]
            },
        ) as mocked:
            interval = get_bybit_funding_interval("btcusdt", category="linear")

        self.assertEqual(interval, timedelta(hours=4))
        mocked.assert_called_once_with(
            "/v5/market/instruments-info",
            {"category": "linear", "symbol": "BTCUSDT"},
        )

    def test_funding_aligns_open_interest_and_account_positioning(self):
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(hours=16)
        timestamp_mid = timestamp_from + timedelta(hours=8)
        responses = [
            {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "fundingRate": "-0.0002",
                        "fundingRateTimestamp": millis(timestamp_mid),
                    },
                    {
                        "symbol": "BTCUSDT",
                        "fundingRate": "0.0001",
                        "fundingRateTimestamp": millis(timestamp_from),
                    },
                ]
            },
            {
                "list": [
                    {
                        "openInterest": "120.5",
                        "singleOpenInterest": "60.25",
                        "timestamp": millis(timestamp_mid),
                    },
                    {
                        "openInterest": "100",
                        "singleOpenInterest": "50",
                        "timestamp": millis(timestamp_from),
                    },
                ],
                "nextPageCursor": "",
            },
            {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "buyRatio": "0.6",
                        "sellRatio": "0.4",
                        "timestamp": millis(timestamp_mid),
                    },
                    {
                        "symbol": "BTCUSDT",
                        "buyRatio": "0.55",
                        "sellRatio": "0.45",
                        "timestamp": millis(timestamp_from),
                    },
                ],
                "nextPageCursor": "",
            },
        ]

        with patch(
            "quant_tick.exchanges.bybit.funding.get_bybit_result",
            side_effect=responses,
        ) as mocked:
            result = bybit_funding(
                "btcusdt",
                timestamp_from,
                timestamp_to,
                category="linear",
            )

        self.assertEqual(list(result.index), [timestamp_from, timestamp_mid])
        first = result.iloc[0]
        self.assertEqual(first.funding_rate, Decimal("0.0001"))
        self.assertEqual(first.open_interest, Decimal(100))
        self.assertEqual(first.single_open_interest, Decimal(50))
        self.assertEqual(first.long_account_ratio, Decimal("0.55"))
        self.assertEqual(first.short_account_ratio, Decimal("0.45"))
        self.assertEqual(
            first.long_short_account_ratio,
            Decimal("0.55") / Decimal("0.45"),
        )
        self.assertEqual(first.open_interest_unit, "base_asset")
        self.assertEqual(first.market_history_interval, "4h")
        self.assertEqual(
            [item.args[0] for item in mocked.call_args_list],
            [
                "/v5/market/funding/history",
                "/v5/market/open-interest",
                "/v5/market/account-ratio",
            ],
        )
        for item in mocked.call_args_list:
            self.assertEqual(item.args[1]["category"], "linear")
            self.assertEqual(item.args[1]["symbol"], "BTCUSDT")

    def test_inverse_funding_uses_inverse_market_history_and_quote_oi(self):
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(hours=8)
        responses = [
            {
                "list": [
                    {
                        "symbol": "BTCUSD",
                        "fundingRate": "0.0001",
                        "fundingRateTimestamp": millis(timestamp_from),
                    }
                ]
            },
            {
                "list": [
                    {
                        "openInterest": "125000000",
                        "singleOpenInterest": "62500000",
                        "timestamp": millis(timestamp_from),
                    }
                ],
                "nextPageCursor": "",
            },
            {
                "list": [
                    {
                        "symbol": "BTCUSD",
                        "buyRatio": "0.6",
                        "sellRatio": "0.4",
                        "timestamp": millis(timestamp_from),
                    }
                ],
                "nextPageCursor": "",
            },
        ]

        with patch(
            "quant_tick.exchanges.bybit.funding.get_bybit_result",
            side_effect=responses,
        ) as mocked:
            result = bybit_funding(
                "BTCUSD",
                timestamp_from,
                timestamp_to,
                category="inverse",
            )

        self.assertEqual(list(result.index), [timestamp_from])
        row = result.iloc[0]
        self.assertEqual(row.open_interest, Decimal(125000000))
        self.assertEqual(row.open_interest_unit, "quote_asset")
        self.assertEqual(row.long_account_ratio, Decimal("0.6"))
        self.assertEqual(row.short_account_ratio, Decimal("0.4"))
        for item in mocked.call_args_list:
            self.assertEqual(item.args[1]["category"], "inverse")
            self.assertEqual(item.args[1]["symbol"], "BTCUSD")

    def test_market_history_absence_does_not_remove_funding(self):
        timestamp_from = datetime(2020, 3, 25, 16, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(hours=8)
        responses = [
            {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "fundingRate": "0.0001",
                        "fundingRateTimestamp": millis(timestamp_from),
                    }
                ]
            },
            {"list": [], "nextPageCursor": ""},
            {"list": [], "nextPageCursor": ""},
        ]

        with patch(
            "quant_tick.exchanges.bybit.funding.get_bybit_result",
            side_effect=responses,
        ):
            result = bybit_funding(
                "BTCUSDT",
                timestamp_from,
                timestamp_to,
                category="linear",
            )

        self.assertEqual(list(result.index), [timestamp_from])
        self.assertEqual(result.iloc[0].funding_rate, Decimal("0.0001"))
        self.assertTrue(pd.isna(result.iloc[0].open_interest))
        self.assertTrue(pd.isna(result.iloc[0].long_account_ratio))

    def test_funding_paginates_newest_to_oldest(self):
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamps = [timestamp_from + timedelta(hours=8 * value) for value in range(3)]
        pages = [
            {
                "list": [
                    {
                        "fundingRateTimestamp": millis(timestamps[2]),
                        "fundingRate": "0.0003",
                    },
                    {
                        "fundingRateTimestamp": millis(timestamps[1]),
                        "fundingRate": "0.0002",
                    },
                ]
            },
            {
                "list": [
                    {
                        "fundingRateTimestamp": millis(timestamps[0]),
                        "fundingRate": "0.0001",
                    }
                ]
            },
        ]

        with (
            patch(
                "quant_tick.exchanges.bybit.funding.FUNDING_MAX_RESULTS",
                2,
            ),
            patch(
                "quant_tick.exchanges.bybit.funding.get_bybit_funding_response",
                side_effect=pages,
            ) as response,
        ):
            rows = _fetch_funding_rows(
                "BTCUSDT",
                timestamp_from,
                timestamps[2],
                category="linear",
            )

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            response.call_args_list,
            [
                call(
                    "BTCUSDT",
                    int(timestamp_from.timestamp() * 1000),
                    int(timestamps[2].timestamp() * 1000),
                    category="linear",
                ),
                call(
                    "BTCUSDT",
                    int(timestamp_from.timestamp() * 1000),
                    int(timestamps[1].timestamp() * 1000) - 1,
                    category="linear",
                ),
            ],
        )

    def test_open_interest_follows_cursor(self):
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(hours=8)
        responses = [
            {
                "list": [
                    {
                        "openInterest": "100",
                        "singleOpenInterest": "50",
                        "timestamp": millis(timestamp_to),
                    }
                ],
                "nextPageCursor": "next",
            },
            {
                "list": [
                    {
                        "openInterest": "90",
                        "singleOpenInterest": "45",
                        "timestamp": millis(timestamp_from),
                    }
                ],
                "nextPageCursor": "",
            },
        ]

        with patch(
            "quant_tick.exchanges.bybit.funding.get_bybit_result",
            side_effect=responses,
        ) as mocked:
            result = bybit_open_interest(
                "BTCUSDT",
                timestamp_from,
                timestamp_to,
                category="linear",
            )

        self.assertEqual(len(result), 2)
        self.assertNotIn("cursor", mocked.call_args_list[0].args[1])
        self.assertEqual(mocked.call_args_list[1].args[1]["cursor"], "next")

    def test_default_funding_cadence_is_eight_hours(self):
        self.assertEqual(
            BybitFunding.expected_timestamps(
                datetime(2026, 4, 25, tzinfo=UTC),
                datetime(2026, 4, 26, tzinfo=UTC),
            ),
            [
                datetime(2026, 4, 25, tzinfo=UTC),
                datetime(2026, 4, 25, 8, tzinfo=UTC),
                datetime(2026, 4, 25, 16, tzinfo=UTC),
            ],
        )

    def test_funding_api_dispatches_bybit_perpetuals(self):
        symbol = SimpleNamespace(
            exchange=Exchange.BYBIT_LINEAR,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
            clamp_timestamp_range=lambda ts_from, ts_to: (ts_from, ts_to),
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 26, tzinfo=UTC)
        expected = pd.DataFrame([])

        with (
            patch(
                "quant_tick.exchanges.api.bybit_funding",
                return_value=expected,
            ) as mocked,
            patch("quant_tick.exchanges.api.refresh_funding_interval") as refresh,
        ):
            result = funding_api(symbol, timestamp_from, timestamp_to)

        refresh.assert_called_once_with(symbol)
        mocked.assert_called_once_with(
            "BTCUSDT",
            timestamp_from,
            timestamp_to,
            category="linear",
            funding_interval=None,
        )
        self.assertTrue(result.equals(expected))

    def test_funding_api_dispatches_bybit_inverse_perpetuals(self):
        symbol = SimpleNamespace(
            exchange=Exchange.BYBIT_INVERSE,
            api_symbol="BTCUSD",
            symbol_type=SymbolType.PERPETUAL,
            clamp_timestamp_range=lambda ts_from, ts_to: (ts_from, ts_to),
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 26, tzinfo=UTC)

        with (
            patch(
                "quant_tick.exchanges.api.bybit_funding",
                return_value=pd.DataFrame([]),
            ) as mocked,
            patch("quant_tick.exchanges.api.refresh_funding_interval") as refresh,
        ):
            funding_api(symbol, timestamp_from, timestamp_to)

        refresh.assert_called_once_with(symbol)
        mocked.assert_called_once_with(
            "BTCUSD",
            timestamp_from,
            timestamp_to,
            category="inverse",
            funding_interval=None,
        )

    def test_configured_funding_cadence_controls_timestamp_normalization(self):
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        funding_timestamp = timestamp_from + timedelta(hours=4)
        timestamp_to = timestamp_from + timedelta(hours=8)
        responses = [
            {
                "list": [
                    {
                        "symbol": "TESTUSDT",
                        "fundingRate": "0.0001",
                        "fundingRateTimestamp": millis(funding_timestamp),
                    }
                ]
            },
            {"list": [], "nextPageCursor": ""},
            {"list": [], "nextPageCursor": ""},
        ]

        with patch(
            "quant_tick.exchanges.bybit.funding.get_bybit_result",
            side_effect=responses,
        ):
            result = bybit_funding(
                "TESTUSDT",
                timestamp_from,
                timestamp_to,
                category="linear",
                funding_interval="4h",
            )

        self.assertEqual(list(result.index), [funding_timestamp])
        self.assertIsNone(result.iloc[0].timestamp_offset_ms)
