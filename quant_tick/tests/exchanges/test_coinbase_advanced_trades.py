from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from quant_tick.exchanges.coinbase_advanced.constants import TRADE_MAX_RESULTS
from quant_tick.exchanges.coinbase_advanced.controllers import (
    CoinbaseAdvancedTrades,
    coinbase_advanced_trades,
)
from quant_tick.exchanges.coinbase_advanced.trades import (
    CoinbaseAdvancedPaginationError,
    get_coinbase_advanced_trades_url,
    get_coinbase_advanced_trades_window,
)


class CoinbaseAdvancedTradesTest(SimpleTestCase):
    def get_trade(
        self,
        trade_id: int,
        timestamp: datetime | None = None,
        side: str = "BUY",
    ) -> dict:
        timestamp = timestamp or datetime(2026, 5, 4, tzinfo=UTC)
        return {
            "trade_id": str(trade_id),
            "time": timestamp.isoformat().replace("+00:00", "Z"),
            "price": "100",
            "size": "1",
            "side": side,
        }

    def test_trades_url_uses_integer_second_window(self):
        url = get_coinbase_advanced_trades_url("BTC-PERP-INTX", 1, 2)
        self.assertEqual(
            url,
            "https://api.coinbase.com/api/v3/brokerage/market/products/"
            "BTC-PERP-INTX/ticker?limit=100&start=1&end=2",
        )

    def test_trades_split_full_integer_second_windows(self):
        full_page = [self.get_trade(index) for index in range(TRADE_MAX_RESULTS)]
        responses = {
            (0, 4): full_page,
            (2, 4): [self.get_trade(4)],
            (0, 2): [self.get_trade(2)],
        }
        calls = []

        def fetch(symbol, timestamp_from, timestamp_to):
            calls.append((timestamp_from, timestamp_to))
            return responses[(timestamp_from, timestamp_to)]

        with patch(
            "quant_tick.exchanges.coinbase_advanced.trades."
            "fetch_coinbase_advanced_trades",
            side_effect=fetch,
        ):
            trades = get_coinbase_advanced_trades_window("BTC-PERP-INTX", 0, 4)

        self.assertEqual(calls, [(0, 4), (2, 4), (0, 2)])
        self.assertEqual([trade["trade_id"] for trade in trades], ["4", "2"])

    def test_trades_raise_when_one_second_page_is_full(self):
        full_page = [self.get_trade(index) for index in range(TRADE_MAX_RESULTS)]

        with (
            patch(
                "quant_tick.exchanges.coinbase_advanced.trades."
                "fetch_coinbase_advanced_trades",
                return_value=full_page,
            ),
            self.assertRaises(CoinbaseAdvancedPaginationError),
        ):
            get_coinbase_advanced_trades_window("BTC-PERP-INTX", 0, 1)

    def test_trades_parse_uppercase_side(self):
        controller = CoinbaseAdvancedTrades.__new__(CoinbaseAdvancedTrades)
        trades = [
            self.get_trade(1, side="BUY"),
            self.get_trade(2, side="SELL"),
        ]

        parsed = controller.parse_data(trades)

        self.assertEqual([trade["tickRule"] for trade in parsed], [-1, 1])

    def test_coinbase_advanced_trades_uses_advanced_controller(self):
        ts_from = datetime(2026, 5, 4, tzinfo=UTC)
        ts_to = datetime(2026, 5, 5, tzinfo=UTC)
        symbol = SimpleNamespace()

        with patch(
            "quant_tick.exchanges.coinbase_advanced.controllers.CoinbaseAdvancedTrades"
        ) as advanced:
            coinbase_advanced_trades(
                symbol,
                ts_from,
                ts_to,
                on_data_frame=lambda *args: None,
            )

        advanced.assert_called_once()
        advanced.return_value.main.assert_called_once()
