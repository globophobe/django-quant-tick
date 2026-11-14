import os
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import httpx2
import pandas as pd
from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase, TestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import exchange_candles, funding, trades_api
from quant_tick.exchanges.femtos.api import get_femtos_response
from quant_tick.exchanges.femtos.candles import femtos_candles
from quant_tick.exchanges.femtos.funding import femtos_funding
from quant_tick.exchanges.femtos.market_history import femtos_market_history
from quant_tick.exchanges.perpetual_stats import perpetual_stats, perpetual_stats_api
from quant_tick.models import ExchangeCandleData, FundingData, PerpetualStatsData

from ..base import BaseSymbolTest

START = datetime(2026, 1, 1, tzinfo=UTC)
END = START + timedelta(hours=2)


def millis(timestamp):
    return int(timestamp.timestamp() * 1000)


def oi(timestamp=START, **values):
    return {
        "t": millis(timestamp),
        "o": "100",
        "h": "130",
        "l": "90",
        "c": "120",
        "px": "70000",
        "usd": "8400000",
        "longAccounts": 8,
        "shortAccounts": 2,
        "longOpen": "25",
        "longClose": "15",
        "shortOpen": "15",
        "shortClose": "5",
        "takerLongOpen": "15",
        "takerShortOpen": "10",
        "liq": "1",
        "trades": 7,
        "funding": "-0.000012345678912345",
        **values,
    }


def cohort(timestamp=START):
    return {
        "t": millis(timestamp) + 23,
        "block": 1234,
        "px": "70000",
        "oi": "100",
        "longAccounts": 6,
        "shortAccounts": 4,
        "top10Long": "40",
        "top10Short": "45",
        "cohorts": [
            {
                "tier": 1,
                "longSz": "20",
                "shortSz": "40",
                "longAccounts": 2,
                "shortAccounts": 3,
            },
            {
                "tier": 4,
                "longSz": "30",
                "shortSz": "10",
                "longAccounts": 4,
                "shortAccounts": 1,
            },
        ],
    }


def candle(timestamp=START, **values):
    return {
        "t": millis(timestamp),
        "o": "70000",
        "h": "70010",
        "l": "69990",
        "c": "70001",
        "v": "1.123456789123456789",
        "vq": "78642.975238641975237889",
        "n": 3,
        **values,
    }


def source(oi_rows=(), cohort_rows=(), candle_rows=(), *, head=END):
    def request(path, params):
        rows = {
            "/hypercore/oi": oi_rows,
            "/hypercore/oi/cohorts": cohort_rows,
            "/candles": candle_rows,
        }[path]
        rows = [row for row in rows if params["from"] <= row["t"] < params["to"]]
        if path == "/candles":
            return {
                "layer": "hypercore",
                "market": params["market"],
                "tf": params["tf"],
                "candles": rows,
                "asOf": millis(head),
            }
        if path == "/hypercore/oi/cohorts":
            return {"coin": params["coin"], "points": rows}
        return {"coin": params["coin"], "tf": "1h", "asOf": millis(head), "bars": rows}

    return request


class FemtosRequestTest(SimpleTestCase):
    def test_key_required_and_header_only(self):
        with (
            patch.dict(os.environ, {"FEMTOS_API_KEY": " "}),
            patch("quant_tick.exchanges.public.httpx2.get") as request,
            self.assertRaises(ImproperlyConfigured),
        ):
            get_femtos_response("/candles", {})
        request.assert_not_called()
        response = httpx2.Response(
            200,
            text='{"value":0.123456789123456789}',
            request=httpx2.Request("GET", "https://api.femtos.xyz/v1/candles"),
        )
        with (
            patch.dict(os.environ, {"FEMTOS_API_KEY": "sample-test-key"}, clear=True),
            patch(
                "quant_tick.exchanges.public.httpx2.get", return_value=response
            ) as request,
        ):
            result = get_femtos_response("/candles", {"market": "BTC"})
        self.assertEqual(result["value"], Decimal("0.123456789123456789"))
        self.assertEqual(
            request.call_args.kwargs["headers"],
            {"Authorization": "Bearer sample-test-key"},
        )
        self.assertEqual(request.call_args.kwargs["params"], {"market": "BTC"})

    def test_auth_and_credit_failures_do_not_retry_or_become_empty_data(self):
        for status in (401, 402, 403):
            response = httpx2.Response(
                status,
                json={"error": "unavailable"},
                request=httpx2.Request("GET", "https://api.femtos.xyz/v1/candles"),
            )
            with (
                self.subTest(status=status),
                patch.dict(
                    os.environ, {"FEMTOS_API_KEY": "sample-test-key"}, clear=True
                ),
                patch(
                    "quant_tick.exchanges.public.httpx2.get", return_value=response
                ) as request,
                self.assertRaises(httpx2.HTTPStatusError),
            ):
                get_femtos_response("/candles", {})
            request.assert_called_once()


class FemtosCandleTest(SimpleTestCase):
    def test_half_open_windows_decimal_units_and_missing_quote_turnover(self):
        end = START + timedelta(minutes=3)
        rows = [
            candle(START + timedelta(minutes=i), vq=None if i == 1 else "123.456")
            for i in range(4)
        ]
        with (
            patch("quant_tick.exchanges.femtos.api.PAGE_SIZE", 2),
            patch("quant_tick.exchanges.femtos.candles.PAGE_SIZE", 2),
            patch(
                "quant_tick.exchanges.femtos.candles.get_femtos_response",
                side_effect=source(candle_rows=rows),
            ) as request,
        ):
            frame = femtos_candles("BTC", START, end)
        self.assertEqual(
            frame.index.tolist(), [START + timedelta(minutes=i) for i in range(3)]
        )
        self.assertEqual(request.call_count, 2)
        self.assertEqual(
            request.call_args_list[1].args[1]["from"],
            millis(START + timedelta(minutes=2)),
        )
        self.assertEqual(frame.notional.iloc[0], Decimal("1.123456789123456789"))
        self.assertEqual(frame.volume.iloc[0], Decimal("123.456"))
        self.assertTrue(pd.isna(frame.volume.iloc[1]))

    def test_missing_and_incomplete_candles_are_not_filled(self):
        end = START + timedelta(minutes=4, seconds=30)
        rows = [candle(START + timedelta(minutes=i)) for i in (0, 2, 3)]
        with patch(
            "quant_tick.exchanges.femtos.candles.get_femtos_response",
            side_effect=source(
                candle_rows=rows, head=START + timedelta(minutes=3, seconds=30)
            ),
        ):
            frame = femtos_candles("BTC", START, end)
        self.assertEqual(frame.index.tolist(), [START, START + timedelta(minutes=2)])

    def test_day_requests_are_split_at_thirty_days_even_across_empty_windows(self):
        end = START + timedelta(days=61)
        with patch(
            "quant_tick.exchanges.femtos.candles.get_femtos_response",
            side_effect=source(),
        ) as request:
            self.assertTrue(femtos_candles("BTC", START, end, "1d").empty)
        self.assertEqual(request.call_count, 3)
        for call in request.call_args_list:
            self.assertLessEqual(
                call.args[1]["to"] - call.args[1]["from"], 30 * 86400000
            )

    def test_wrong_market_duplicate_and_invalid_ohlc_fail(self):
        good = source(candle_rows=[candle()])(
            "/candles",
            {"market": "BTC", "tf": "1m", "from": millis(START), "to": millis(END)},
        )
        cases = [
            dict(good, market="ETH"),
            dict(good, candles=[candle(), candle()]),
            dict(good, candles=[candle(l="80000")]),
        ]
        for result in cases:
            with (
                self.subTest(result=result),
                patch(
                    "quant_tick.exchanges.femtos.candles.get_femtos_response",
                    return_value=result,
                ),
                self.assertRaises(ValueError),
            ):
                femtos_candles("BTC", START, END)

    def test_unsupported_resolution_and_spot_stats_are_rejected_before_request(self):
        symbol = SimpleNamespace(
            exchange=Exchange.HYPERLIQUID, api_symbol="BTC", symbol_type=SymbolType.SPOT
        )
        with patch(
            "quant_tick.exchanges.femtos.candles.get_femtos_response"
        ) as request:
            with self.assertRaises(ValueError):
                femtos_candles("BTC", START, END, "7m")
            with self.assertRaises(ValueError):
                perpetual_stats_api(symbol, START, END)
        request.assert_not_called()


class FemtosStatsTest(SimpleTestCase):
    def collect(self, oi_rows, cohort_rows, *, start=START, end=END, head=END):
        respond = source(oi_rows, cohort_rows, head=head)
        with (
            patch(
                "quant_tick.exchanges.femtos.api.get_femtos_response",
                side_effect=respond,
            ),
            patch(
                "quant_tick.exchanges.femtos.market_history.get_femtos_response",
                side_effect=respond,
            ),
        ):
            return femtos_market_history("BTC", start, end)

    def test_cohorts_and_interval_end_oi_keep_their_distinct_meanings(self):
        row = self.collect([oi()], [cohort()]).iloc[0]
        self.assertEqual(row.open_interest, Decimal(120))
        self.assertEqual(row.single_open_interest, Decimal(60))
        self.assertEqual(row.open_interest_value, Decimal(8400000))
        self.assertEqual(row.long_account_ratio, Decimal("0.6"))
        self.assertEqual(row.long_short_account_ratio, Decimal("1.5"))
        self.assertEqual(row.long_accounts, Decimal(6))
        self.assertEqual(row.oi_long_accounts, Decimal(8))
        self.assertEqual(row.large_long_size, Decimal(30))
        self.assertEqual(row.large_short_size, Decimal(10))
        self.assertEqual(row.cohort_timestamp, START + timedelta(milliseconds=23))
        self.assertEqual(row.funding_rate, Decimal("-0.000012345678912345"))
        self.assertEqual(row.cohorts, cohort()["cohorts"])

    def test_missing_cohort_keeps_hour_retryable_and_old_oi_remains_available(self):
        second = START + timedelta(hours=1)
        frame = self.collect([oi(), oi(second)], [cohort()])
        self.assertEqual(frame.index.tolist(), [START])
        old = datetime(2024, 1, 1, tzinfo=UTC)
        row = self.collect(
            [oi(old, longOpen=None, longAccounts=None, shortAccounts=None)],
            [],
            start=old,
            end=old + timedelta(hours=1),
        ).iloc[0]
        self.assertEqual(row.open_interest, Decimal(120))
        self.assertIsNone(row.long_open_size)
        self.assertNotIn("long_account_ratio", row.index)

    def test_unfinished_oi_and_inconsistent_cohorts_are_rejected(self):
        self.assertTrue(
            self.collect([oi()], [cohort()], head=START + timedelta(minutes=30)).empty
        )
        for field in ("longAccounts", "oi"):
            point = deepcopy(cohort())
            point[field] = 999
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "reconcile"),
            ):
                self.collect([oi()], [point])

    def test_funding_uses_settlement_hour_and_preserves_negative_rates(self):
        rows = [oi(START - timedelta(hours=1)), oi(), oi(START + timedelta(hours=1))]
        with patch(
            "quant_tick.exchanges.femtos.api.get_femtos_response",
            side_effect=source(oi_rows=rows),
        ) as request:
            frame = femtos_funding("BTC", START, START + timedelta(hours=1, minutes=30))
        self.assertEqual(frame.index.tolist(), [START, START + timedelta(hours=1)])
        self.assertEqual(
            frame.funding_rate.tolist(), [Decimal("-0.000012345678912345")] * 2
        )
        self.assertEqual(
            request.call_args.args[1]["from"], millis(START - timedelta(hours=1))
        )

    def test_unsettled_and_distributed_eight_hour_funding_are_absent(self):
        with patch(
            "quant_tick.exchanges.femtos.api.get_femtos_response",
            side_effect=source(oi_rows=[oi(funding=None)]),
        ):
            self.assertTrue(
                femtos_funding("BTC", START + timedelta(hours=1), END).empty
            )
        with patch("quant_tick.exchanges.femtos.api.get_femtos_response") as request:
            self.assertTrue(
                femtos_funding(
                    "BTC",
                    datetime(2023, 6, 1, tzinfo=UTC),
                    datetime(2023, 6, 2, tzinfo=UTC),
                ).empty
            )
        request.assert_not_called()


@patch.dict(os.environ, {"FEMTOS_API_KEY": "test-key", "COINAPI_API_KEY": ""})
class FemtosCollectionTest(BaseSymbolTest, TestCase):
    def setUp(self):
        super().setUp()
        self.symbol = self.get_symbol(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
            exchange_candle_resolution="1m",
        )

    def test_missing_hour_is_repaired_and_complete_history_skips_requests(self):
        second = START + timedelta(hours=1)
        points = [cohort()]
        respond = source([oi(), oi(second)], points)
        with (
            patch(
                "quant_tick.exchanges.femtos.api.get_femtos_response",
                side_effect=respond,
            ) as oi_request,
            patch(
                "quant_tick.exchanges.femtos.market_history.get_femtos_response",
                side_effect=respond,
            ),
        ):
            perpetual_stats(self.symbol, START, END)
            self.assertEqual(PerpetualStatsData.objects.count(), 1)
            points.append(cohort(second))
            perpetual_stats(self.symbol, START, END)
            self.assertEqual(PerpetualStatsData.objects.count(), 2)
            oi_request.reset_mock()
            perpetual_stats(self.symbol, START, END)
            oi_request.assert_not_called()
        row = PerpetualStatsData.objects.first()
        self.assertEqual(row.long_account_ratio, Decimal("0.6"))
        self.assertEqual(row.json_data["large_long_size"], Decimal(30))
        self.assertEqual(row.json_data["cohorts"][1]["longSz"], Decimal(30))
        self.assertEqual(row.json_data["market_history_source"], "femtos")
        self.assertEqual(row.symbol.exchange, Exchange.HYPERLIQUID)

    def test_femtos_key_does_not_replace_native_candles_or_funding(self):
        candle_frame = pd.DataFrame(
            [
                {
                    "timestamp": START,
                    "open": Decimal(70000),
                    "high": Decimal(70010),
                    "low": Decimal(69990),
                    "close": Decimal(70000),
                    "notional": Decimal("1.123456789123456789"),
                }
            ]
        ).set_index("timestamp")
        funding_frame = pd.DataFrame(
            {
                "funding_rate": [Decimal("0.0001"), Decimal("0.0002")],
            },
            index=[START, START + timedelta(hours=1)],
        )
        with (
            patch(
                "quant_tick.exchanges.api.hyperliquid_funding",
                return_value=funding_frame,
            ),
            patch(
                "quant_tick.exchanges.api.hyperliquid_candles",
                return_value=candle_frame,
            ),
            patch("quant_tick.exchanges.femtos.api.get_femtos_response") as provider,
        ):
            funding(self.symbol, START, END)
            exchange_candles(self.symbol, START, START + timedelta(minutes=1))
        provider.assert_not_called()
        self.assertEqual(FundingData.objects.count(), 2)
        row = ExchangeCandleData.objects.get()
        # SQLite stores Decimal fields through floating point; parsing is exact above.
        self.assertAlmostEqual(row.notional, Decimal("1.123456789123456789"), places=14)
        self.assertIsNone(row.volume)
        self.symbol.refresh_from_db()
        self.assertEqual(self.symbol.funding_interval, timedelta(hours=1))

    def test_femtos_key_alone_keeps_native_trade_collection(self):
        with (
            patch("quant_tick.exchanges.public.httpx2.get") as request,
            patch(
                "quant_tick.exchanges.hyperliquid.controllers.HyperliquidTradesWebSocket"
            ) as native,
        ):
            trades_api(self.symbol, START, END, lambda *args: None)
        request.assert_not_called()
        native.return_value.main.assert_called_once()

    def test_missing_key_skips_positioning_without_writing_placeholder_rows(self):
        with (
            patch.dict(os.environ, {"FEMTOS_API_KEY": " "}),
            patch("quant_tick.exchanges.femtos.api.get_femtos_response") as provider,
        ):
            perpetual_stats(self.symbol, START, END)
        provider.assert_not_called()
        self.assertFalse(PerpetualStatsData.objects.filter(symbol=self.symbol).exists())
