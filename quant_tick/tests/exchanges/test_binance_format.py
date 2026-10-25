from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

from django.test import SimpleTestCase, TestCase, override_settings

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import api, candles_api, trades_api
from quant_tick.exchanges.aster.controllers import AsterTrades
from quant_tick.exchanges.binance_format import raw_trades
from quant_tick.exchanges.binance_us.controllers import BinanceUSTrades
from quant_tick.models import TradeData

from ..base import BaseSymbolTest

START = datetime(2025, 1, 1, tzinfo=UTC)
MS = int(START.timestamp() * 1000)


def trade(trade_id, timestamp=MS, price="100", quantity="0.01"):
    return {
        "id": trade_id,
        "time": timestamp,
        "price": price,
        "qty": quantity,
        "isBuyerMaker": trade_id % 2 == 0,
        "quoteQty": "0",
    }


class RawTradeHistoryTest(SimpleTestCase):
    def test_aster_uses_first_trade_one_for_empty_seek_hours_and_history_end(self):
        first = trade(1)

        def response(path, params):
            if path == "/aggTrades" or params.get("fromId") == 0:
                return []
            return [first]

        controller = AsterTrades(
            Mock(api_symbol="BTCUSDT"), START, START + timedelta(minutes=1), Mock()
        )
        with patch.object(AsterTrades, "request", side_effect=response):
            rows, _, _ = controller.iter_api(START, START + timedelta(minutes=1))
            self.assertFalse(
                controller.is_history_exhausted(START + timedelta(minutes=1))
            )
            self.assertTrue(controller.is_history_exhausted(START))
        self.assertEqual([row["id"] for row in rows], [1])

    def test_aster_collects_short_pages_and_crosses_empty_id_ranges(self):
        history = [
            trade(1, MS - 1),
            trade(2),
            trade(4),
            trade(1001),
            trade(3003),
            trade(3004, MS + 60000),
        ]

        def response(path, params):
            if path == "/aggTrades":
                return [{"f": 1, "T": MS - 1}]
            if "fromId" not in params:
                return history[-1:]
            start = params["fromId"]
            return [
                row for row in history if start <= row["id"] < start + params["limit"]
            ]

        controller = AsterTrades(
            Mock(api_symbol="BTCUSDT"), START, START + timedelta(minutes=1), Mock()
        )
        with patch.object(AsterTrades, "request", side_effect=response):
            rows, finished, cursor = controller.iter_api(
                START, START + timedelta(minutes=1)
            )
        self.assertEqual([row["id"] for row in rows], [3003, 1001, 4, 2])
        self.assertTrue(finished)
        self.assertIsNone(cursor)

    def test_aster_orders_timestamps_and_reads_across_the_time_boundary(self):
        history = [
            trade(1, MS - 1),
            trade(2, MS + 22),
            trade(3, MS),
            trade(999, MS + 60000),
            trade(1001, MS + 59980),
            trade(1999, MS + 60020),
            trade(2001, MS + 60050),
            trade(5000, MS + 120000),
        ]
        cursors = []

        def response(path, params):
            if path == "/aggTrades":
                return [{"f": 1, "T": MS - 1}]
            if "fromId" not in params:
                return history[-1:]
            start = params["fromId"]
            cursors.append(start)
            return [
                row for row in history if start <= row["id"] < start + params["limit"]
            ]

        controller = AsterTrades(
            Mock(api_symbol="BTCUSDT"), START, START + timedelta(minutes=1), Mock()
        )
        with patch.object(AsterTrades, "request", side_effect=response):
            rows, finished, cursor = controller.iter_api(
                START, START + timedelta(minutes=1)
            )
        self.assertEqual([row["id"] for row in rows], [1001, 2, 3])
        self.assertEqual([row["time"] for row in rows], [MS + 59980, MS + 22, MS])
        self.assertEqual(cursors, [1, 1000, 2000])
        self.assertTrue(finished)
        self.assertIsNone(cursor)

    def test_aster_quiet_seek_hour_can_land_on_a_missing_id_range(self):
        history = [
            trade(i, MS - 7200000 if i < 3000 else MS) for i in (1, 1500, 3002, 4003)
        ]

        def response(path, params):
            if path == "/aggTrades":
                return []
            if "fromId" not in params:
                return history[-1:]
            start = params["fromId"]
            return [
                row for row in history if start <= row["id"] < start + params["limit"]
            ]

        controller = AsterTrades(
            Mock(api_symbol="BTCUSDT"), START, START + timedelta(minutes=1), Mock()
        )
        with patch.object(AsterTrades, "request", side_effect=response):
            rows, _, _ = controller.iter_api(START, START + timedelta(minutes=1))
        self.assertEqual([row["id"] for row in rows], [4003, 3002])

    def test_anchor_overlap_and_equal_timestamp_page_boundary(self):
        request = Mock(
            side_effect=[
                [{"f": 0, "T": MS - 1}],
                [trade(0, MS - 1), *[trade(i) for i in range(1, 1000)]],
                [trade(1000, price="110"), trade(1001, MS + 60000)],
            ]
        )
        rows, finished, cursor = raw_trades(
            request, "BTCUSDT", START, START + timedelta(minutes=1)
        )
        self.assertTrue(finished)
        self.assertIsNone(cursor)
        self.assertEqual([row["id"] for row in rows], list(range(1000, 0, -1)))
        self.assertEqual(request.call_args_list[1].args[1]["fromId"], 0)
        self.assertEqual(request.call_args_list[2].args[1]["fromId"], 1000)

    def test_first_trade_is_retained_without_an_aggregate_anchor(self):
        request = Mock(side_effect=[[], [trade(0)], [trade(0), trade(1)]])
        rows, _, _ = raw_trades(request, "BTCUSDT", START, START + timedelta(minutes=1))
        self.assertEqual([row["id"] for row in rows], [1, 0])

    def test_repeated_or_reordered_raw_ids_fail_for_each_venue(self):
        for controller_type in (AsterTrades, BinanceUSTrades):
            for second in (trade(1), trade(0)):
                with self.subTest(controller=controller_type, second=second):

                    def response(path, params, second=second):
                        if path == "/aggTrades":
                            return [{"f": 1, "T": MS - 1}]
                        if "fromId" not in params:
                            return [trade(3)]
                        return [trade(1), second]

                    controller = controller_type(
                        Mock(api_symbol="BTCUSDT"),
                        START,
                        START + timedelta(minutes=1),
                        Mock(),
                    )
                    with (
                        patch.object(controller_type, "request", side_effect=response),
                        self.assertRaisesRegex(ValueError, "ordered by ID"),
                    ):
                        controller.iter_api(START, START + timedelta(minutes=1))

    def test_binance_us_still_rejects_timestamp_regressions(self):
        request = Mock(
            side_effect=[[{"f": 1, "T": MS - 1}], [trade(1), trade(2, MS - 1)]]
        )
        with self.assertRaisesRegex(ValueError, "ordered"):
            raw_trades(request, "BTCUSDT", START, START + timedelta(minutes=1))

    def test_quiet_seek_hour_uses_raw_ids_without_replaying_full_history(self):
        history = [trade(i, MS - 7200000 if i < 4 else MS) for i in range(7)]

        def response(path, params):
            if path == "/aggTrades":
                return []
            if "fromId" not in params:
                return history[-1:]
            return history[params["fromId"] :][: params["limit"]]

        request = Mock(side_effect=response)
        rows, _, _ = raw_trades(request, "BTCUSDT", START, START + timedelta(minutes=1))
        self.assertEqual([row["id"] for row in rows], [6, 5, 4])
        self.assertLess(request.call_count, 10)


@override_settings(
    STORAGES={"default": {"BACKEND": "django.core.files.storage.InMemoryStorage"}}
)
class BinanceFormatCollectionTest(BaseSymbolTest, TestCase):
    def test_individual_trades_persist_and_validate_for_each_venue(self):
        cases = (
            (Exchange.BINANCE_US, SymbolType.SPOT, "0.03", True),
            (Exchange.ASTER, SymbolType.PERPETUAL, "0.03", True),
            (Exchange.ASTER, SymbolType.PERPETUAL, "0.04", False),
        )
        for exchange, symbol_type, candle_quantity, expected_ok in cases:
            with self.subTest(exchange=exchange, candle_quantity=candle_quantity):
                symbol = self.get_symbol(
                    exchange=exchange, symbol_type=symbol_type, api_symbol="BTCUSDT"
                )
                calls = []

                def response(
                    base,
                    path,
                    params,
                    calls=calls,
                    exchange=exchange,
                    candle_quantity=candle_quantity,
                    **kwargs,
                ):
                    calls.append((base, path, params))
                    if path == "/klines":
                        if params["endTime"] < MS:
                            return []
                        return [[MS, "100", "110", "100", "110", candle_quantity]]
                    if path == "/aggTrades":
                        return [{"f": 9, "T": MS - 1}]
                    if path == "/historicalTrades":
                        last_id = 13 if exchange == Exchange.ASTER else 11
                        history = [
                            trade(9, MS - 1),
                            trade(10),
                            trade(last_id, MS + 22, price="110", quantity="0.02"),
                        ]
                        if exchange == Exchange.ASTER:
                            history[1] = trade(
                                10, MS + 22, price="110", quantity="0.02"
                            )
                            history[2] = trade(last_id)
                        if "fromId" not in params:
                            return history[-1:]
                        start = params["fromId"]
                        return [
                            row
                            for row in history
                            if start <= row["id"] < start + params["limit"]
                        ]
                    raise AssertionError(path)

                with patch(
                    f"quant_tick.exchanges.{exchange.replace('-', '_')}.controllers.get_public_json",
                    side_effect=response,
                ):
                    api(symbol, START, START + timedelta(minutes=1))
                stored = TradeData.objects.get(symbol=symbol)
                self.assertIs(stored.ok, expected_ok)
                frame = stored.get_data_frame("raw_data")
                self.assertEqual(
                    stored.uid, "10" if exchange == Exchange.BINANCE_US else "13"
                )
                self.assertEqual(len(frame), 2)
                self.assertEqual(
                    frame.timestamp.tolist(),
                    [START, START + timedelta(milliseconds=22)],
                )
                self.assertEqual(frame.price.tolist(), [Decimal(100), Decimal(110)])
                self.assertEqual(frame.volume.sum(), Decimal("3.2"))
                self.assertEqual(
                    frame.tickRule.tolist(),
                    [-1, 1] if exchange == Exchange.BINANCE_US else [1, -1],
                )
                self.assertTrue(
                    all(
                        exchange.replace("-", ".") in base
                        if exchange == Exchange.BINANCE_US
                        else "asterdex.com" in base
                        for base, _, _ in calls
                    )
                )

    def test_candles_dispatch_and_reject_wrong_market_type(self):
        for exchange, symbol_type, prefix in (
            (Exchange.BINANCE_US, SymbolType.SPOT, "binance_us"),
            (Exchange.ASTER, SymbolType.PERPETUAL, "aster"),
        ):
            with self.subTest(exchange=exchange):
                symbol = self.get_symbol(
                    exchange=exchange, symbol_type=symbol_type, api_symbol="BTCUSDT"
                )
                with patch(f"quant_tick.exchanges.api.{prefix}_candles") as fetch:
                    candles_api(
                        symbol, START, START + timedelta(hours=8), resolution="8h"
                    )
                self.assertEqual(fetch.call_args.kwargs["resolution"], "8h")
                symbol.symbol_type = (
                    SymbolType.PERPETUAL
                    if symbol_type == SymbolType.SPOT
                    else SymbolType.SPOT
                )
                with self.assertRaises(ValueError):
                    trades_api(symbol, START, START + timedelta(hours=1), Mock())
                with self.assertRaises(ValueError):
                    candles_api(symbol, START, START + timedelta(hours=1))

    def test_backfill_stops_before_first_trade_but_not_on_quiet_hour(self):
        symbol = self.get_symbol(exchange=Exchange.BINANCE_US, api_symbol="BTCUSDT")
        first = trade(0, MS + 3600000)
        candles = Mock(empty=True)
        windows = [
            (START + timedelta(hours=h), START + timedelta(hours=h + 1))
            for h in (2, 1, 0, -1)
        ]

        def response(path, params):
            if path == "/aggTrades":
                return []
            return [first]

        received = Mock()
        with (
            patch(
                "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                return_value=iter(windows),
            ),
            patch.object(BinanceUSTrades, "request", side_effect=response),
            patch.object(BinanceUSTrades, "get_candles", return_value=candles),
            patch.object(
                BinanceUSTrades, "validate_websocket_partitions", return_value={}
            ),
        ):
            BinanceUSTrades(
                symbol, START - timedelta(days=1), START + timedelta(hours=3), received
            ).main()
        self.assertEqual(received.call_count, 3)
        self.assertTrue(received.call_args_list[0].args[3].empty)
        self.assertEqual(received.call_args_list[1].args[3].uid.tolist(), ["0"])
