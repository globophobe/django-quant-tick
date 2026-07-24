from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import SimpleTestCase

from quant_tick.constants import SymbolType
from quant_tick.exchanges.deribit.controllers import DeribitTrades, deribit_trades
from quant_tick.exchanges.deribit.trades import get_trades


class DeribitTradesTest(SimpleTestCase):
    def setUp(self):
        self.timestamp_from = datetime(2026, 7, 24, tzinfo=UTC)

    def test_get_trades_pages_backwards_by_sequence(self):
        trades = [
            {"trade_seq": 102, "timestamp": 1784851201000},
            {"trade_seq": 101, "timestamp": 1784851201000},
        ]
        pagination_id = {"end_timestamp": 1784851201999}

        with patch(
            "quant_tick.exchanges.deribit.trades.get_deribit_trades_response",
            return_value={"trades": trades, "has_more": True},
        ) as mocked:
            result, is_last, next_pagination_id = get_trades(
                "BTC-PERPETUAL",
                self.timestamp_from,
                pagination_id,
            )

        mocked.assert_called_once_with(
            "BTC-PERPETUAL",
            int(self.timestamp_from.timestamp() * 1000),
            pagination_id,
        )
        self.assertEqual(result, trades)
        self.assertFalse(is_last)
        self.assertEqual(next_pagination_id, {"end_seq": 100})

    def test_get_trades_ends_when_response_has_no_more(self):
        with patch(
            "quant_tick.exchanges.deribit.trades.get_deribit_trades_response",
            return_value={
                "trades": [{"trade_seq": 7, "timestamp": 1784851201000}],
                "has_more": False,
            },
        ):
            _, is_last, next_pagination_id = get_trades(
                "BTC-PERPETUAL",
                self.timestamp_from,
                {"end_seq": 7},
            )

        self.assertTrue(is_last)
        self.assertEqual(next_pagination_id, {"end_seq": 6})

    def test_mixin_maps_inverse_perpetual_units_and_ordering(self):
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
                "trade_seq": 101,
                "trade_id": "BTC-1",
                "timestamp": 1784851201000,
                "direction": "buy",
                "amount": 250,
                "price": 62500.0,
            },
        ]

        parsed = controller.parse_data(raw)
        frame = controller.get_data_frame(parsed)

        self.assertEqual(frame["uid"].tolist(), ["101", "102"])
        self.assertEqual(frame["index"].tolist(), [101, 102])
        self.assertEqual(frame["volume"].tolist(), [Decimal("250"), Decimal("500")])
        self.assertEqual(
            frame["notional"].tolist(),
            [Decimal("0.004"), Decimal("0.008")],
        )
        self.assertEqual(frame["tickRule"].tolist(), [1, -1])

    def test_mixin_uses_sequence_before_next_stored_partition(self):
        controller = DeribitTrades.__new__(DeribitTrades)
        controller.symbol = SimpleNamespace()

        with patch(
            "quant_tick.exchanges.deribit.base.TradeData.objects.get_last_uid",
            return_value="101",
        ) as mocked:
            result = controller.get_pagination_id(self.timestamp_from)

        mocked.assert_called_once_with(controller.symbol, self.timestamp_from)
        self.assertEqual(result, {"end_seq": 100})

    def test_mixin_uses_half_open_timestamp_for_initial_page(self):
        controller = DeribitTrades.__new__(DeribitTrades)
        controller.symbol = SimpleNamespace()

        with patch(
            "quant_tick.exchanges.deribit.base.TradeData.objects.get_last_uid",
            return_value=None,
        ):
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
