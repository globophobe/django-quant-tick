from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from django.test import SimpleTestCase

from quant_tick.constants import SymbolType
from quant_tick.exchanges.deribit.controllers import DeribitTrades, deribit_trades
from quant_tick.exchanges.deribit.trades import get_trades


class DeribitTradesTest(SimpleTestCase):
    def setUp(self):
        self.timestamp_from = datetime(2026, 7, 24, tzinfo=UTC)

    @staticmethod
    def trade(sequence: int, timestamp: int) -> dict:
        return {
            "trade_seq": sequence,
            "trade_id": f"BTC-{sequence}",
            "instrument_name": "BTC-PERPETUAL",
            "timestamp": timestamp,
            "direction": "buy",
            "amount": 100,
            "price": 62500.0,
        }

    def test_get_trades_fetches_exact_half_open_history_window(self):
        timestamp_to = self.timestamp_from + timedelta(milliseconds=4)
        start = int(self.timestamp_from.timestamp() * 1000)
        trades = [self.trade(100, start + 1)]

        with patch(
            "quant_tick.exchanges.deribit.trades.get_deribit_trades_response",
            return_value={"trades": trades, "has_more": False},
        ) as mocked:
            result = get_trades(
                "BTC-PERPETUAL",
                self.timestamp_from,
                timestamp_to,
                history=True,
            )

        mocked.assert_called_once_with(
            "BTC-PERPETUAL",
            start,
            start + 3,
            history=True,
        )
        self.assertEqual(result, (trades, True, None))

    def test_get_trades_bisects_overflowing_window(self):
        timestamp_to = self.timestamp_from + timedelta(milliseconds=4)
        start = int(self.timestamp_from.timestamp() * 1000)
        first = self.trade(100, start + 1)
        second = self.trade(102, start + 2)

        with patch(
            "quant_tick.exchanges.deribit.trades.get_deribit_trades_response",
            side_effect=[
                {"trades": [first], "has_more": True},
                {"trades": [first], "has_more": False},
                {"trades": [second], "has_more": False},
            ],
        ) as mocked:
            result, is_last, pagination_id = get_trades(
                "BTC-PERPETUAL",
                self.timestamp_from,
                timestamp_to,
                history=True,
            )

        self.assertEqual(
            mocked.call_args_list,
            [
                call("BTC-PERPETUAL", start, start + 3, history=True),
                call("BTC-PERPETUAL", start, start + 1, history=True),
                call("BTC-PERPETUAL", start + 2, start + 3, history=True),
            ],
        )
        self.assertEqual(result, [first, second])
        self.assertTrue(is_last)
        self.assertIsNone(pagination_id)

    def test_get_trades_deduplicates_only_identical_payloads(self):
        timestamp_to = self.timestamp_from + timedelta(milliseconds=1)
        start = int(self.timestamp_from.timestamp() * 1000)
        trade = self.trade(100, start)

        with patch(
            "quant_tick.exchanges.deribit.trades.get_deribit_trades_response",
            return_value={"trades": [trade, trade.copy()], "has_more": False},
        ):
            result, _, _ = get_trades(
                "BTC-PERPETUAL",
                self.timestamp_from,
                timestamp_to,
                history=True,
            )

        self.assertEqual(result, [trade])

        conflict = trade | {"amount": 200}
        with patch(
            "quant_tick.exchanges.deribit.trades.get_deribit_trades_response",
            return_value={"trades": [trade, conflict], "has_more": False},
        ):
            with self.assertRaisesRegex(ValueError, "conflicting trade sequence 100"):
                get_trades(
                    "BTC-PERPETUAL",
                    self.timestamp_from,
                    timestamp_to,
                    history=True,
                )

    def test_get_trades_selects_recent_and_history_apis(self):
        timestamp_to = self.timestamp_from + timedelta(milliseconds=1)
        start = int(self.timestamp_from.timestamp() * 1000)
        trade = self.trade(100, start)

        for current_time, expected_history in (
            (self.timestamp_from + timedelta(hours=1), False),
            (self.timestamp_from + timedelta(days=2), True),
        ):
            with (
                patch(
                    "quant_tick.exchanges.deribit.trades.get_current_time",
                    return_value=current_time,
                ),
                patch(
                    "quant_tick.exchanges.deribit.trades.get_deribit_trades_response",
                    return_value={"trades": [trade], "has_more": False},
                ) as mocked,
            ):
                get_trades(
                    "BTC-PERPETUAL",
                    self.timestamp_from,
                    timestamp_to,
                )

            mocked.assert_called_once_with(
                "BTC-PERPETUAL",
                start,
                start,
                history=expected_history,
            )
    def test_get_trades_rejects_invalid_window_responses(self):
        timestamp_to = self.timestamp_from + timedelta(milliseconds=1)
        start = int(self.timestamp_from.timestamp() * 1000)

        outside = self.trade(100, start + 1)
        with patch(
            "quant_tick.exchanges.deribit.trades.get_deribit_trades_response",
            return_value={"trades": [outside], "has_more": False},
        ):
            with self.assertRaisesRegex(ValueError, "outside requested window"):
                get_trades(
                    "BTC-PERPETUAL",
                    self.timestamp_from,
                    timestamp_to,
                    history=True,
                )

        inside = self.trade(100, start)
        with patch(
            "quant_tick.exchanges.deribit.trades.get_deribit_trades_response",
            return_value={"trades": [inside], "has_more": True},
        ):
            with self.assertRaisesRegex(ValueError, "within one millisecond"):
                get_trades(
                    "BTC-PERPETUAL",
                    self.timestamp_from,
                    timestamp_to,
                    history=True,
                )

    def test_mixin_maps_inverse_perpetual_units_and_sparse_ordering(self):
        controller = DeribitTrades.__new__(DeribitTrades)
        controller.symbol = SimpleNamespace(
            symbol_type=SymbolType.PERPETUAL,
            api_symbol="BTC-PERPETUAL",
        )
        raw = [
            {
                "trade_seq": 102,
                "trade_id": "BTC-2",
                "timestamp": 1784851201000,
                "direction": "sell",
                "amount": 500,
                "price": 62500.0,
            },
            {
                "trade_seq": 100,
                "trade_id": "BTC-1",
                "timestamp": 1784851201000,
                "direction": "buy",
                "amount": 250,
                "price": 62500.0,
            },
        ]
        raw.reverse()

        parsed = controller.parse_data(raw)
        frame = controller.get_data_frame(parsed)
        controller.assert_data_frame(
            self.timestamp_from,
            self.timestamp_from + timedelta(minutes=1),
            frame,
            parsed,
        )

        self.assertEqual(frame["uid"].tolist(), ["100", "102"])
        self.assertEqual(frame["index"].tolist(), [100, 102])
        self.assertEqual(frame["volume"].tolist(), [Decimal("250"), Decimal("500")])
        self.assertEqual(
            frame["notional"].tolist(),
            [Decimal("0.004"), Decimal("0.008")],
        )
        self.assertEqual(frame["tickRule"].tolist(), [1, -1])

    def test_mixin_uses_half_open_timestamp_window(self):
        controller = DeribitTrades.__new__(DeribitTrades)
        result = controller.get_pagination_id(self.timestamp_from)

        self.assertEqual(
            result,
            {"end_timestamp": int(self.timestamp_from.timestamp() * 1000) - 1},
        )

    def test_mixin_fetches_one_minute_candles_for_validation(self):
        controller = DeribitTrades.__new__(DeribitTrades)
        controller.symbol = SimpleNamespace(api_symbol="BTC-PERPETUAL")
        timestamp_to = self.timestamp_from + timedelta(minutes=1)

        with patch(
            "quant_tick.exchanges.deribit.base.deribit_candles",
            return_value=Mock(),
        ) as mocked:
            controller.get_candles(self.timestamp_from, timestamp_to)

        mocked.assert_called_once_with(
            "BTC-PERPETUAL",
            self.timestamp_from,
            timestamp_to,
            resolution="1m",
        )

    def test_main_fetches_each_missing_partition_independently(self):
        controller = DeribitTrades.__new__(DeribitTrades)
        controller.symbol = SimpleNamespace()
        controller.timestamp_from = self.timestamp_from
        controller.timestamp_to = self.timestamp_from + timedelta(hours=2)
        controller.retry = False
        controller.get_candles = Mock(side_effect=["newer", "older"])
        controller.validate_websocket_partitions = Mock(return_value={})
        controller.get_websocket_timestamp_from = Mock(
            return_value=controller.timestamp_to
        )
        controller.on_rest_data_frame = Mock()
        newer = (
            self.timestamp_from + timedelta(hours=1),
            self.timestamp_from + timedelta(hours=2),
        )
        older = (self.timestamp_from, self.timestamp_from + timedelta(hours=1))

        with patch(
            "quant_tick.controllers.rest.TradeDataIterator.iter_all",
            return_value=[newer, older],
        ):
            controller.main()

        self.assertEqual(
            controller.on_rest_data_frame.call_args_list,
            [call(*newer, "newer"), call(*older, "older")],
        )

    def test_deribit_trades_uses_exchange_controller(self):
        timestamp_to = self.timestamp_from + timedelta(minutes=1)
        symbol = SimpleNamespace(symbol_type=SymbolType.PERPETUAL)

        with patch(
            "quant_tick.exchanges.deribit.controllers.DeribitTrades"
        ) as controller:
            deribit_trades(
                symbol,
                self.timestamp_from,
                timestamp_to,
                on_data_frame=Mock(),
            )

        controller.assert_called_once()
        controller.return_value.main.assert_called_once_with()
    def test_mixin_maps_spot_amount_as_base_asset(self):
        controller = DeribitTrades.__new__(DeribitTrades)
        controller.symbol = SimpleNamespace(
            symbol_type=SymbolType.SPOT,
            api_symbol="BTC_USDC",
        )
        parsed = controller.parse_data(
            [
                {
                    "trade_seq": 1,
                    "timestamp": 1784851201000,
                    "direction": "buy",
                    "amount": 0.004,
                    "price": 62500.0,
                }
            ]
        )

        self.assertEqual(parsed[0]["notional"], Decimal("0.004"))
        self.assertEqual(parsed[0]["volume"], Decimal("250.0000"))

    def test_mixin_maps_linear_perpetual_amount_as_base_asset(self):
        controller = DeribitTrades.__new__(DeribitTrades)
        controller.symbol = SimpleNamespace(
            symbol_type=SymbolType.PERPETUAL,
            api_symbol="BTC_USDC-PERPETUAL",
        )
        parsed = controller.parse_data(
            [
                {
                    "trade_seq": 1,
                    "timestamp": 1784851201000,
                    "direction": "sell",
                    "amount": 0.004,
                    "price": 62500.0,
                }
            ]
        )

        self.assertEqual(parsed[0]["notional"], Decimal("0.004"))
        self.assertEqual(parsed[0]["volume"], Decimal("250.0000"))

    def test_mixin_maps_option_amount_and_starbase_timestamp(self):
        controller = DeribitTrades.__new__(DeribitTrades)
        controller.symbol = SimpleNamespace(
            symbol_type=SymbolType.PERPETUAL,
            api_symbol="BTC-24APR26-72000-C",
        )
        parsed = controller.parse_data(
            [
                {
                    "trade_seq": 1,
                    "timestamp": 1785110400000,
                    "starbase_timestamp": 1785110400123456789,
                    "direction": "buy",
                    "amount": 3,
                    "price": 0.0525,
                }
            ]
        )

        self.assertEqual(parsed[0]["notional"], Decimal("3"))
        self.assertEqual(parsed[0]["volume"], Decimal("0.1575"))
        self.assertEqual(
            parsed[0]["timestamp"],
            datetime(2026, 7, 27, 0, 0, 0, 123456, tzinfo=UTC),
        )
        self.assertEqual(parsed[0]["nanoseconds"], 789)
