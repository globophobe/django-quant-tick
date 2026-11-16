import json
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx2
import pandas as pd
from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase, TestCase, override_settings

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import api, candles_api, exchange_candles, trades_api
from quant_tick.exchanges.coinapi.api import get_coinapi_response
from quant_tick.exchanges.coinapi.candles import coinapi_candles
from quant_tick.exchanges.coinapi.controllers import CoinAPITrades
from quant_tick.models import ExchangeCandleData, TradeData

from ..base import BaseSymbolTest

START = datetime(2025, 1, 1, tzinfo=UTC)
END = START + timedelta(minutes=1)
SYMBOL = "BINANCE_SPOT_BTC_USDT"
HYPERLIQUID_SYMBOL = "HYPERLIQUID_PERP_BTC_USDC"


def trade(uid, timestamp="2025-01-01T00:00:00.0000001Z", side="BUY", **kwargs):
    return {
        "symbol_id": SYMBOL,
        "uuid": str(uid),
        "time_exchange": timestamp,
        "time_coinapi": "2025-01-01T00:00:01.0000001Z",
        "price": "100.123456789123456789",
        "size": "0.1",
        "taker_side": side,
        **kwargs,
    }


def candle(timestamp=START, minutes=1, quantity="0.2", count=2):
    return {
        "time_period_start": timestamp.isoformat(),
        "time_period_end": (timestamp + timedelta(minutes=minutes)).isoformat(),
        "price_open": "100.123456789123456789",
        "price_high": "100.123456789123456789",
        "price_low": "100.123456789123456789",
        "price_close": "100.123456789123456789",
        "volume_traded": quantity,
        "trades_count": count,
    }


def response(status, payload):
    return httpx2.Response(
        status,
        text=json.dumps(payload),
        request=httpx2.Request("GET", "https://rest.coinapi.io/v1/trades/test/history"),
    )


class CoinAPIRequestTest(SimpleTestCase):
    def test_all_rest_endpoints_share_the_configured_requests_per_minute(self):
        def check_rate_limit(configured, rate_limit):
            now = 1000.0
            started = []
            latencies = (0.1, 3.0, 7.0) * 7
            durations = iter(latencies)

            def advance(seconds):
                nonlocal now
                now += seconds

            def respond(*args, **kwargs):
                started.append(now)
                advance(next(durations))
                return response(200, [])

            environ = {"COINAPI_API_KEY": "test-key"}
            if configured is not None:
                environ["COINAPI_RATE_LIMIT"] = configured
            with (
                patch.dict(os.environ, environ, clear=True),
                patch("quant_tick.exchanges.public.time.time", side_effect=lambda: now),
                patch("quant_tick.exchanges.public.time.sleep", side_effect=advance),
                patch("quant_tick.exchanges.public.httpx2.get", side_effect=respond),
            ):
                for path in (
                    "/v1/trades/test/history",
                    "/v1/ohlcv/test/history",
                    "/v1/symbols",
                ) * 7:
                    self.assertEqual(get_coinapi_response(path, {}), [])

            self.assertEqual(len(started), 21)
            for (earlier, later), latency in zip(pairwise(started), latencies):
                self.assertAlmostEqual(later - earlier, max(60 / rate_limit, latency))
            for start in started:
                self.assertLessEqual(
                    sum(start <= value < start + 60 - 1e-8 for value in started),
                    rate_limit,
                )

        for configured, rate_limit in (
            (None, 10),
            ("", 10),
            ("40", 40),
            ("160", 160),
            ("17", 17),
        ):
            with self.subTest(configured=configured):
                check_rate_limit(configured, rate_limit)

    def test_invalid_rate_limit_fails_before_any_request(self):
        for value in ("0", "-10", "1.5", "nan", "inf", "invalid"):
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ,
                    {"COINAPI_API_KEY": "test-key", "COINAPI_RATE_LIMIT": value},
                    clear=True,
                ),
                patch("quant_tick.exchanges.public.httpx2.get") as request,
                self.assertRaisesRegex(ImproperlyConfigured, "COINAPI_RATE_LIMIT"),
            ):
                get_coinapi_response("/v1/trades/test/history", {})
            request.assert_not_called()

    def test_key_required_before_any_request(self):
        with (
            patch.dict(os.environ, {"COINAPI_API_KEY": "  "}),
            patch("quant_tick.exchanges.public.httpx2.get") as request,
            self.assertRaisesRegex(ImproperlyConfigured, "COINAPI_API_KEY"),
        ):
            get_coinapi_response("/v1/trades/test/history", {})
        request.assert_not_called()

    def test_key_is_header_only_and_numeric_precision_is_preserved(self):
        with (
            patch.dict(os.environ, {"COINAPI_API_KEY": "sample-test-key"}, clear=True),
            patch(
                "quant_tick.exchanges.public.httpx2.get",
                return_value=httpx2.Response(200, text="[0.123456789123456789]"),
            ) as request,
        ):
            request.return_value.request = httpx2.Request(
                "GET", "https://rest.coinapi.io"
            )
            self.assertEqual(
                get_coinapi_response("/v1/trades/test/history", {"limit": 1}),
                [Decimal("0.123456789123456789")],
            )
        request.assert_called_once_with(
            "https://rest.coinapi.io/v1/trades/test/history",
            params={"limit": 1},
            timeout=30,
            headers={"X-CoinAPI-Key": "sample-test-key"},
        )

    def test_auth_and_plan_errors_are_not_retried_or_returned_as_empty_data(self):
        for status in (401, 402, 403):
            with (
                self.subTest(status=status),
                patch.dict(
                    os.environ, {"COINAPI_API_KEY": "sample-test-key"}, clear=True
                ),
                patch(
                    "quant_tick.exchanges.public.httpx2.get",
                    return_value=response(status, {}),
                ) as request,
                self.assertRaises(httpx2.HTTPStatusError),
            ):
                get_coinapi_response("/v1/trades/test/history", {})
            request.assert_called_once()

    def test_error_object_is_not_treated_as_market_data(self):
        with (
            patch.dict(os.environ, {"COINAPI_API_KEY": "sample-test-key"}, clear=True),
            patch(
                "quant_tick.exchanges.public.httpx2.get",
                return_value=response(200, {"error": "unavailable"}),
            ),
            self.assertRaisesRegex(TypeError, "unexpected"),
        ):
            get_coinapi_response("/v1/trades/test/history", {})


class CoinAPIMarketTest(SimpleTestCase):
    def controller(self):
        return CoinAPITrades(SimpleNamespace(api_symbol=SYMBOL), START, END, Mock())

    def test_complete_window_preserves_delayed_trades_ties_and_nanoseconds(self):
        first = trade(1, timestamp=START.isoformat())
        second = trade(2, side="SELL", id_trade="venue-2")
        third = trade(3)
        fourth = trade(4, timestamp="2025-01-01T00:00:00.0000002Z")
        outside = trade(5, timestamp=END.isoformat())
        controller = self.controller()
        with (
            patch("quant_tick.exchanges.coinapi.controllers.PAGE_SIZE", 2),
            patch(
                "quant_tick.exchanges.coinapi.controllers.get_coinapi_response",
                side_effect=[
                    [first, fourth],
                    [first, fourth, second, third, outside],
                ],
            ) as request,
        ):
            frame = controller.fetch_rest_data_frame(START, END)
        self.assertEqual(frame.uid.tolist(), ["1", "2", "3", "4"])
        self.assertEqual(frame.tickRule.tolist(), [1, -1, 1, 1])
        self.assertEqual(frame.nanoseconds.tolist(), [0, 100, 100, 200])
        self.assertEqual(
            frame.timestamp.tolist(),
            [
                pd.Timestamp(row["time_exchange"])
                for row in (first, second, third, fourth)
            ],
        )
        self.assertEqual(frame.volume.sum(), Decimal("40.0493827156493827156"))
        self.assertEqual(frame.id_trade.dropna().tolist(), ["venue-2"])
        self.assertEqual(frame.id_trade.isna().tolist(), [True, False, True, True])
        self.assertEqual(
            request.call_args_list[1].args[1]["time_start"], START.isoformat()
        )
        self.assertEqual(request.call_args_list[1].args[1]["time_end"], END.isoformat())
        self.assertEqual(request.call_args_list[1].args[1]["limit"], 20)
        self.assertEqual(request.call_count, 2)

    def test_unpageable_timestamp_or_outside_window_trades_fail(self):
        pages = (
            [
                trade(1, timestamp=START.isoformat()),
                trade(2, timestamp=START.isoformat()),
            ],
            [trade(1, timestamp="2025-01-01T00:01:00.0000001Z")],
            [trade(1, timestamp="2024-12-31T23:59:59Z")],
        )
        for page in pages:
            with (
                self.subTest(page=page),
                patch("quant_tick.exchanges.coinapi.controllers.PAGE_SIZE", 2),
                patch("quant_tick.exchanges.coinapi.controllers.MAX_PAGE_SIZE", 2),
                patch(
                    "quant_tick.exchanges.coinapi.controllers.get_coinapi_response",
                    return_value=page,
                ),
                self.assertRaises(ValueError),
            ):
                self.controller().fetch_rest_data_frame(START, END)

    def test_repeated_page_fails_instead_of_losing_or_duplicating_trades(self):
        page = [
            trade(1, timestamp="2025-01-01T00:00:10Z"),
            trade(2, timestamp="2025-01-01T00:00:40Z"),
        ]
        with (
            patch("quant_tick.exchanges.coinapi.controllers.PAGE_SIZE", 2),
            patch("quant_tick.exchanges.coinapi.controllers.MAX_PAGE_SIZE", 2),
            patch(
                "quant_tick.exchanges.coinapi.controllers.get_coinapi_response",
                return_value=page,
            ),
            self.assertRaisesRegex(ValueError, "outside the requested window"),
        ):
            self.controller().fetch_rest_data_frame(START, END)

    def test_capped_windows_split_without_losing_delayed_or_boundary_trades(self):
        rows = [
            trade("first", timestamp=START.isoformat()),
            trade("last", timestamp="2025-01-01T00:00:59Z"),
            trade("late", timestamp="2025-01-01T00:00:29.9999999Z"),
            trade("boundary", timestamp="2025-01-01T00:00:30Z"),
            trade("middle", timestamp="2025-01-01T00:00:40Z"),
            trade("outside", timestamp=END.isoformat()),
        ]

        def history(path, params):
            start, end = (
                pd.Timestamp(params["time_start"]),
                pd.Timestamp(params["time_end"]),
            )
            return [
                row
                for row in rows
                if start <= pd.Timestamp(row["time_exchange"]) <= end
            ][: params["limit"]]

        with (
            patch("quant_tick.exchanges.coinapi.controllers.PAGE_SIZE", 3),
            patch("quant_tick.exchanges.coinapi.controllers.MAX_PAGE_SIZE", 3),
            patch(
                "quant_tick.exchanges.coinapi.controllers.get_coinapi_response",
                side_effect=history,
            ),
        ):
            frame = self.controller().fetch_rest_data_frame(START, END)
        self.assertEqual(
            frame.uid.tolist(), ["first", "late", "boundary", "middle", "last"]
        )
        self.assertEqual(frame.notional.sum(), Decimal("0.5"))
        self.assertTrue(frame.timestamp.is_monotonic_increasing)

    def test_split_windows_cannot_drop_trades_seen_in_capped_parent(self):
        first = trade("first", timestamp=START.isoformat())
        last = trade("last", timestamp="2025-01-01T00:00:59Z")
        with (
            patch("quant_tick.exchanges.coinapi.controllers.PAGE_SIZE", 2),
            patch("quant_tick.exchanges.coinapi.controllers.MAX_PAGE_SIZE", 2),
            patch(
                "quant_tick.exchanges.coinapi.controllers.get_coinapi_response",
                side_effect=[[first, last], [], [last]],
            ),
            self.assertRaisesRegex(ValueError, "omitted.*observed"),
        ):
            self.controller().fetch_rest_data_frame(START, END)

    def test_replayed_venue_trade_preserves_first_occurrence_and_closing_price(self):
        first = trade("first", id_trade="venue-1", price="100", size="0.2")
        second = trade("second", id_trade="venue-2", price="101", size="0.3")
        replay = {
            **first,
            "uuid": "replay",
            "time_coinapi": "2025-01-01T00:00:30Z",
        }
        with patch(
            "quant_tick.exchanges.coinapi.controllers.get_coinapi_response",
            return_value=[first, second, replay],
        ):
            frame = self.controller().fetch_rest_data_frame(START, END)
        self.assertEqual(frame.uid.tolist(), ["first", "second"])
        self.assertEqual(frame["index"].tolist(), [0, 1])
        self.assertEqual(frame.notional.sum(), Decimal("0.5"))
        self.assertEqual(frame.price.iloc[-1], Decimal(101))

    def test_venue_trade_replay_requires_matching_values_and_exchange_identity(self):
        first = trade("first", id_trade="venue-1")
        for changes in ({"price": "101"}, {"size": "0.2"}, {"taker_side": "SELL"}):
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(ValueError, "exchange trade ID has conflicting"),
            ):
                self.controller().parse_data(
                    [{**first, "uuid": "replay", **changes}, first]
                )
        self.assertEqual(len(self.controller().parse_data([trade(1), trade(2)])), 2)
        later = {**first, "uuid": "later", "time_exchange": "2025-01-01T00:00:10Z"}
        self.assertEqual(len(self.controller().parse_data([later, first])), 2)

    def test_unreported_sides_and_invalid_trade_values_fail(self):
        invalid = [
            trade(1, side=side)
            for side in ("BUY_ESTIMATED", "SELL_ESTIMATED", "UNKNOWN", "")
        ]
        invalid += [
            trade(1, **values)
            for values in (
                {"symbol_id": "COINBASE_SPOT_BTC_USD"},
                {"size": "-1"},
                {"price": "NaN"},
                {"price": "0"},
                {"size": "Infinity"},
                {"uuid": ""},
            )
        ]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.controller().parse_data([row])
        with self.assertRaisesRegex(ValueError, "duplicated"):
            self.controller().parse_data([trade(1), trade(1)])

    def test_complete_window_cannot_drop_observed_trades(self):
        first = trade(1, timestamp=START.isoformat())
        boundary = trade(2)
        for next_page in ([], [trade(3, timestamp="2025-01-01T00:00:00.0000002Z")]):
            with (
                self.subTest(next_page=next_page),
                patch("quant_tick.exchanges.coinapi.controllers.PAGE_SIZE", 2),
                patch(
                    "quant_tick.exchanges.coinapi.controllers.get_coinapi_response",
                    side_effect=[[first, boundary], next_page],
                ),
                self.assertRaisesRegex(ValueError, "omitted.*observed"),
            ):
                self.controller().fetch_rest_data_frame(START, END)

    def test_candle_pagination_uses_period_end_and_base_quantity(self):
        with (
            patch("quant_tick.exchanges.coinapi.candles.PAGE_SIZE", 2),
            patch(
                "quant_tick.exchanges.coinapi.candles.get_coinapi_response",
                side_effect=[
                    [candle(), candle(END)],
                    [candle(END + timedelta(minutes=1))],
                ],
            ) as request,
        ):
            frame = coinapi_candles(SYMBOL, START, START + timedelta(minutes=3))
        self.assertEqual(frame.index.tolist(), [START, END, END + timedelta(minutes=1)])
        self.assertEqual(frame.notional.sum(), Decimal("0.6"))
        self.assertNotIn("volume", frame.columns)
        self.assertEqual(
            pd.Timestamp(request.call_args.args[1]["time_start"]),
            START + timedelta(minutes=2),
        )

    def test_quiet_candle_response_outside_window_and_partial_candles_are_excluded(
        self,
    ):
        with patch(
            "quant_tick.exchanges.coinapi.candles.get_coinapi_response",
            return_value=[candle(START - timedelta(minutes=1)), candle(END)],
        ):
            self.assertTrue(
                coinapi_candles(SYMBOL, START, END + timedelta(seconds=30)).empty
            )

    def test_candle_replay_cannot_loop_or_duplicate_rows(self):
        with (
            patch("quant_tick.exchanges.coinapi.candles.PAGE_SIZE", 2),
            patch(
                "quant_tick.exchanges.coinapi.candles.get_coinapi_response",
                return_value=[candle(), candle(END)],
            ),
            self.assertRaisesRegex(ValueError, "pagination did not advance"),
        ):
            coinapi_candles(SYMBOL, START, START + timedelta(minutes=3))


@override_settings(
    STORAGES={"default": {"BACKEND": "django.core.files.storage.InMemoryStorage"}}
)
@patch(
    "quant_tick.exchanges.coinapi.flat_files.CoinAPITradeFiles.get_trades",
    Mock(return_value=None),
)
class CoinAPICollectionTest(BaseSymbolTest, TestCase):
    def test_backfill_preserves_partial_missing_hours_on_both_routes(self):
        end = START + timedelta(hours=2)
        candles = [
            candle(START + timedelta(minutes=i), quantity="0.1") for i in range(120)
        ]
        cases = (
            (15, 75, [(60, 75), (15, 60)]),
            (45, 75, [(60, 75), (45, 60)]),
            (15, 45, [(15, 45)]),
        )
        for exchange, api_symbol in (
            (Exchange.COINAPI, HYPERLIQUID_SYMBOL),
            (Exchange.HYPERLIQUID, "BTC"),
        ):
            for first, last, intervals in cases:
                with self.subTest(exchange=exchange, first=first, last=last):
                    symbol = self.get_symbol(
                        exchange=exchange,
                        api_symbol=api_symbol,
                        symbol_type=SymbolType.PERPETUAL,
                    )
                    symbol.code_name = f"partial-{symbol.pk}"
                    symbol.save(update_fields=["code_name"])
                    existing = TradeData.objects.bulk_create(
                        [
                            TradeData(
                                symbol=symbol,
                                timestamp=START + timedelta(minutes=i),
                                frequency=1,
                                ok=True,
                            )
                            for i in range(120)
                            if not first <= i < last
                        ]
                    )

                    def archive(start, stop):
                        self.assertEqual(
                            pd.Timestamp(start).floor("h"),
                            (pd.Timestamp(stop) - pd.Timedelta("1ns")).floor("h"),
                        )
                        return [
                            trade(
                                i,
                                timestamp=(START + timedelta(minutes=i)).isoformat(),
                                symbol_id=HYPERLIQUID_SYMBOL,
                                id_trade=str(i),
                            )
                            for i in range(120)
                            if start <= START + timedelta(minutes=i) < stop
                        ]

                    with (
                        patch.dict(os.environ, {"COINAPI_API_KEY": "test-key"}),
                        patch(
                            "quant_tick.exchanges.coinapi.flat_files.CoinAPITradeFiles.get_trades",
                            side_effect=archive,
                        ) as files,
                        patch(
                            "quant_tick.exchanges.coinapi.candles.get_coinapi_response",
                            return_value=candles,
                        ) as requests,
                        patch(
                            "quant_tick.exchanges.hyperliquid.controllers.native_hyperliquid_candles",
                            return_value=pd.DataFrame(
                                index=pd.DatetimeIndex([], tz="UTC")
                            ),
                        ),
                        patch(
                            "quant_tick.exchanges.coinapi.api._coinapi_history_start",
                            return_value=pd.Timestamp(START),
                        ),
                        patch(
                            "quant_tick.exchanges.coinapi.controllers.get_coinapi_response"
                        ) as rest,
                    ):
                        api(symbol, START, end)
                        self.assertEqual(
                            [call.args for call in files.call_args_list],
                            [
                                (
                                    START + timedelta(minutes=start),
                                    START + timedelta(minutes=stop),
                                )
                                for start, stop in intervals
                            ],
                        )
                        requests.assert_called_once()
                        params = requests.call_args.args[1]
                        self.assertEqual(
                            pd.Timestamp(params["time_start"]),
                            START + timedelta(minutes=first),
                        )
                        self.assertEqual(
                            pd.Timestamp(params["time_end"]),
                            START + timedelta(minutes=last),
                        )
                        rest.assert_not_called()
                        stored = list(
                            TradeData.objects.filter(symbol=symbol).order_by(
                                "timestamp"
                            )
                        )
                        self.assertEqual(
                            [
                                row.timestamp + timedelta(minutes=i)
                                for row in stored
                                for i in range(row.frequency)
                            ],
                            [START + timedelta(minutes=i) for i in range(120)],
                        )
                        self.assertTrue(all(row.ok for row in stored))
                        self.assertTrue(
                            {row.pk for row in existing} <= {row.pk for row in stored}
                        )
                        frames = pd.concat(
                            [
                                row.get_data_frame("raw_data")
                                for row in stored
                                if row.raw_data
                            ],
                            ignore_index=True,
                        )
                        self.assertEqual(
                            frames.timestamp.tolist(),
                            [START + timedelta(minutes=i) for i in range(first, last)],
                        )
                        self.assertEqual(
                            frames.notional.sum(), Decimal("0.1") * (last - first)
                        )
                        files.reset_mock()
                        requests.reset_mock()
                        api(symbol, START, end)
                        files.assert_not_called()
                        requests.assert_not_called()
                        rest.assert_not_called()

    def test_backfill_batches_candles_and_preserves_hourly_progress_on_both_routes(
        self,
    ):
        end = START + timedelta(hours=2)
        rows = [
            trade(
                minute,
                timestamp=(START + timedelta(minutes=minute)).isoformat(),
                symbol_id=HYPERLIQUID_SYMBOL,
                id_trade=str(minute),
            )
            for minute in range(120)
        ]
        candles = [
            candle(START + timedelta(minutes=i), quantity="0.1") for i in range(120)
        ]

        def archive(start, end):
            return [
                row for row in rows if start <= pd.Timestamp(row["time_exchange"]) < end
            ]

        def interrupted_archive(start, end):
            if start == START:
                raise RuntimeError("Archive temporarily unavailable")
            return archive(start, end)

        for exchange, api_symbol in (
            (Exchange.COINAPI, HYPERLIQUID_SYMBOL),
            (Exchange.HYPERLIQUID, "BTC"),
        ):
            symbol = self.get_symbol(
                exchange=exchange,
                api_symbol=api_symbol,
                symbol_type=SymbolType.PERPETUAL,
            )
            with (
                self.subTest(exchange=exchange),
                patch.dict(os.environ, {"COINAPI_API_KEY": "test-key"}),
                patch(
                    "quant_tick.exchanges.coinapi.flat_files.CoinAPITradeFiles.get_trades",
                    side_effect=interrupted_archive,
                ) as files,
                patch(
                    "quant_tick.exchanges.coinapi.candles.get_coinapi_response",
                    return_value=candles,
                ) as requests,
                patch(
                    "quant_tick.exchanges.hyperliquid.controllers.native_hyperliquid_candles",
                    return_value=pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC")),
                ),
                patch(
                    "quant_tick.exchanges.coinapi.api._coinapi_history_start",
                    return_value=pd.Timestamp(START),
                ),
                patch(
                    "quant_tick.exchanges.coinapi.controllers.get_coinapi_response"
                ) as rest,
            ):
                with self.assertRaisesRegex(RuntimeError, "temporarily unavailable"):
                    api(symbol, START, end)
                requests.assert_called_once()
                self.assertEqual(
                    pd.Timestamp(requests.call_args.args[1]["time_start"]), START
                )
                self.assertEqual(
                    pd.Timestamp(requests.call_args.args[1]["time_end"]), end
                )
                self.assertEqual(files.call_count, 2)
                rest.assert_not_called()
                saved = TradeData.objects.get(symbol=symbol)
                self.assertEqual(saved.timestamp, START + timedelta(hours=1))
                self.assertTrue(saved.ok)
                requests.reset_mock()
                files.reset_mock()
                files.side_effect = archive
                api(symbol, START, end)
                requests.assert_called_once()
                self.assertEqual(
                    pd.Timestamp(requests.call_args.args[1]["time_end"]),
                    START + timedelta(hours=1),
                )
                files.assert_called_once()
                stored = list(
                    TradeData.objects.filter(symbol=symbol).order_by("timestamp")
                )
                self.assertEqual([row.frequency for row in stored], [60, 60])
                self.assertEqual([row.ok for row in stored], [True, True])
                self.assertEqual(
                    [row.get_data_frame("raw_data").notional.sum() for row in stored],
                    [Decimal(6)] * 2,
                )
                requests.reset_mock()
                files.reset_mock()
                api(symbol, START, end)
                requests.assert_not_called()
                files.assert_not_called()

    def test_ingestion_persists_validated_raw_aggregated_and_filtered_data(self):
        for symbol_id, symbol_type, native_exchange, native_pair in (
            (SYMBOL, SymbolType.SPOT, Exchange.BINANCE, "BTCUSDT"),
            (HYPERLIQUID_SYMBOL, SymbolType.PERPETUAL, Exchange.HYPERLIQUID, "BTC"),
        ):
            with self.subTest(symbol_id=symbol_id):
                symbol = self.get_symbol(
                    exchange=Exchange.COINAPI,
                    api_symbol=symbol_id,
                    symbol_type=symbol_type,
                    save_aggregated=True,
                    significant_trade_filter=1,
                )
                native = self.get_symbol(
                    exchange=native_exchange,
                    api_symbol=native_pair,
                    symbol_type=symbol_type,
                )
                rows = [
                    trade(1, symbol_id=symbol_id, id_trade="1065044209871623"),
                    trade(2, symbol_id=symbol_id, side="SELL", id_trade="11"),
                    trade(3, symbol_id=symbol_id, id_trade="1065044209871623"),
                ]
                with (
                    patch.dict(
                        os.environ, {"COINAPI_API_KEY": "sample-test-key"}, clear=True
                    ),
                    patch(
                        "quant_tick.exchanges.public.httpx2.get",
                        side_effect=[
                            response(200, [candle(quantity="0.3", count=3)]),
                            response(200, rows),
                        ],
                    ) as request,
                ):
                    api(symbol, START, END)
                self.assertEqual(request.call_count, 2)
                self.assertEqual(
                    request.call_args.args[0],
                    "https://rest.coinapi.io/v1/trades/" + symbol_id + "/history",
                )
                stored = TradeData.objects.get(symbol=symbol)
                self.assertIs(stored.ok, True)
                for field in ("raw_data", "aggregated_data", "filtered_data"):
                    frame = stored.get_data_frame(field)
                    self.assertEqual(frame.notional.sum(), Decimal("0.2"))
                    self.assertEqual(
                        frame.volume.sum(), Decimal("20.0246913578246913578")
                    )
                    self.assertEqual(frame.tickRule.tolist(), [1, -1])
                    self.assertEqual(frame.nanoseconds.tolist(), [100, 100])
                raw = stored.get_data_frame("raw_data")
                self.assertEqual(raw.coinapi_uuid.tolist(), ["1", "2"])
                self.assertEqual(raw.id_trade.tolist(), ["1065044209871623", "11"])
                self.assertEqual(
                    raw.coinapi_replayed_notional.tolist(), [Decimal("0.1"), 0]
                )
                self.assertEqual(
                    raw.time_coinapi.tolist(), [rows[0]["time_coinapi"]] * 2
                )
                self.assertNotEqual(symbol.upload_path, native.upload_path)
                self.assertFalse(TradeData.objects.filter(symbol=native).exists())

    def test_unknown_sides_abort_before_partition_write(self):
        symbol = self.get_symbol(exchange=Exchange.COINAPI, api_symbol=SYMBOL)
        with (
            patch(
                "quant_tick.exchanges.coinapi.candles.get_coinapi_response",
                return_value=[candle()],
            ),
            patch(
                "quant_tick.exchanges.coinapi.controllers.get_coinapi_response",
                return_value=[trade(1, side="UNKNOWN")],
            ),
            self.assertRaisesRegex(ValueError, "reported"),
        ):
            api(symbol, START, END)
        self.assertFalse(TradeData.objects.filter(symbol=symbol).exists())

    def test_missing_or_mismatched_candles_do_not_validate_trades(self):
        for candles, expected in (([], None), ([candle(quantity="1")], False)):
            symbol = self.get_symbol(exchange=Exchange.COINAPI, api_symbol=SYMBOL)
            with (
                self.subTest(expected=expected),
                patch(
                    "quant_tick.exchanges.coinapi.candles.get_coinapi_response",
                    return_value=candles,
                ),
                patch(
                    "quant_tick.exchanges.coinapi.controllers.get_coinapi_response",
                    return_value=[trade(1)],
                ),
            ):
                api(symbol, START, END)
            self.assertIs(TradeData.objects.get(symbol=symbol).ok, expected)

    def test_larger_candles_use_hourly_source_and_preserve_volume_units(self):
        symbol = self.get_symbol(
            exchange=Exchange.COINAPI,
            api_symbol=HYPERLIQUID_SYMBOL,
            symbol_type=SymbolType.PERPETUAL,
            exchange_candle_resolution="8h",
        )
        with patch(
            "quant_tick.exchanges.coinapi.candles.get_coinapi_response",
            return_value=[
                candle(START + timedelta(hours=hour), minutes=60)
                for hour in (0, 3, 7, 8)
            ],
        ) as request:
            exchange_candles(symbol, START, START + timedelta(hours=9))
        self.assertEqual(request.call_args.args[1]["period_id"], "1HRS")
        self.assertEqual(
            request.call_args.args[0],
            "/v1/ohlcv/HYPERLIQUID_PERP_BTC_USDC/history",
        )
        stored = ExchangeCandleData.objects.get(symbol=symbol)
        self.assertEqual(stored.frequency, 480)
        self.assertEqual(stored.notional, Decimal("0.6"))
        self.assertIsNone(stored.volume)

    def test_dispatch_rejects_unsupported_or_mistyped_markets_before_requesting(self):
        for symbol_id, symbol_type in (
            (SYMBOL, SymbolType.PERPETUAL),
            (HYPERLIQUID_SYMBOL, SymbolType.SPOT),
            ("BINANCEFTS_PERP_BTC_USDT", SymbolType.SPOT),
            ("BINANCEFTS_PERP_BTC_USDT", SymbolType.PERPETUAL),
            ("HYPERLIQUIDL4_PERP_BTC_USDC", SymbolType.PERPETUAL),
            ("HYPERLIQUID_PERP_BTC_USD", SymbolType.PERPETUAL),
            ("HYPERLIQUID_PERP_BTC_USDC_EXTRA", SymbolType.PERPETUAL),
            ("BYBIT_PERP_BTC_USDT", SymbolType.SPOT),
            ("BYBIT_PERP_BTC_USD", SymbolType.PERPETUAL),
            ("BYBIT_PERP_BTC_USDC", SymbolType.PERPETUAL),
            ("BTCUSDT", SymbolType.SPOT),
        ):
            symbol = self.get_symbol(
                exchange=Exchange.COINAPI, api_symbol=symbol_id, symbol_type=symbol_type
            )
            with patch("quant_tick.exchanges.public.httpx2.get") as request:
                for collect, args in (
                    (trades_api, (symbol, START, END, Mock())),
                    (candles_api, (symbol, START, END)),
                ):
                    with (
                        self.subTest(symbol_id=symbol_id, collect=collect),
                        self.assertRaises(ValueError),
                    ):
                        collect(*args)
            request.assert_not_called()
