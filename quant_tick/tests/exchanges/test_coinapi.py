import json
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
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


def candle(timestamp=START, minutes=1, quantity="0.2"):
    return {
        "time_period_start": timestamp.isoformat(),
        "time_period_end": (timestamp + timedelta(minutes=minutes)).isoformat(),
        "price_open": "100.123456789123456789",
        "price_high": "100.123456789123456789",
        "price_low": "100.123456789123456789",
        "price_close": "100.123456789123456789",
        "volume_traded": quantity,
        "trades_count": 2,
    }


def response(status, payload):
    return httpx2.Response(
        status,
        text=json.dumps(payload),
        request=httpx2.Request("GET", "https://rest.coinapi.io/v1/trades/test/history"),
    )


class CoinAPIRequestTest(SimpleTestCase):
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

    def test_time_cursor_preserves_tied_trades_and_nanoseconds(self):
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
                    [first, second],
                    [second, third],
                    [second, third, fourth, outside],
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
            request.call_args_list[1].args[1]["time_start"], second["time_exchange"]
        )
        self.assertEqual(request.call_args_list[2].args[1]["limit"], 20)

    def test_unpageable_timestamp_or_disordered_trades_fail(self):
        pages = (
            [
                trade(1, timestamp=START.isoformat()),
                trade(2, timestamp=START.isoformat()),
            ],
            [trade(1, timestamp="2025-01-01T00:00:10Z"), trade(2)],
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
        page = [trade(1, timestamp=START.isoformat()), trade(2)]
        with (
            patch("quant_tick.exchanges.coinapi.controllers.PAGE_SIZE", 2),
            patch(
                "quant_tick.exchanges.coinapi.controllers.get_coinapi_response",
                return_value=page,
            ),
            self.assertRaisesRegex(ValueError, "unordered or outside"),
        ):
            self.controller().fetch_rest_data_frame(START, END)

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

    def test_page_cannot_silently_drop_its_shared_timestamp(self):
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
                self.assertRaisesRegex(ValueError, "omitted.*boundary"),
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
class CoinAPICollectionTest(BaseSymbolTest, TestCase):
    def test_ingestion_persists_validated_raw_aggregated_and_filtered_data(self):
        symbol = self.get_symbol(
            exchange=Exchange.COINAPI,
            api_symbol=SYMBOL,
            save_aggregated=True,
            significant_trade_filter=1,
        )
        native = self.get_symbol(exchange=Exchange.BINANCE, api_symbol="BTCUSDT")
        rows = [trade(1, id_trade="10"), trade(2, side="SELL", id_trade="11")]
        with (
            patch.dict(os.environ, {"COINAPI_API_KEY": "sample-test-key"}, clear=True),
            patch(
                "quant_tick.exchanges.public.httpx2.get",
                side_effect=[response(200, [candle()]), response(200, rows)],
            ) as request,
        ):
            api(symbol, START, END)
        self.assertEqual(request.call_count, 2)
        stored = TradeData.objects.get(symbol=symbol)
        self.assertIs(stored.ok, True)
        for field in ("raw_data", "aggregated_data", "filtered_data"):
            frame = stored.get_data_frame(field)
            self.assertEqual(frame.notional.sum(), Decimal("0.2"))
            self.assertEqual(frame.volume.sum(), Decimal("20.0246913578246913578"))
            self.assertEqual(frame.tickRule.tolist(), [1, -1])
            self.assertEqual(frame.nanoseconds.tolist(), [100, 100])
        raw = stored.get_data_frame("raw_data")
        self.assertEqual(raw.coinapi_uuid.tolist(), ["1", "2"])
        self.assertEqual(raw.id_trade.tolist(), ["10", "11"])
        self.assertEqual(raw.time_coinapi.tolist(), [rows[0]["time_coinapi"]] * 2)
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
            api_symbol=SYMBOL,
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
        stored = ExchangeCandleData.objects.get(symbol=symbol)
        self.assertEqual(stored.frequency, 480)
        self.assertEqual(stored.notional, Decimal("0.6"))
        self.assertIsNone(stored.volume)

    def test_dispatch_rejects_nonspot_or_native_symbols_before_requesting(self):
        for symbol_id, symbol_type in (
            (SYMBOL, SymbolType.PERPETUAL),
            ("BINANCEFTS_PERP_BTC_USDT", SymbolType.SPOT),
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
