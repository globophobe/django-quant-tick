from datetime import UTC, datetime
from decimal import Decimal

from django.test import TestCase

from quant_tick.constants import Exchange
from quant_tick.models import Symbol, WebSocketData


class WebSocketDataTest(TestCase):
    def get_trade(self, uid: str = "ws-1") -> dict:
        return {
            "uid": uid,
            "timestamp": "2026-05-10T10:00:10Z",
            "nanoseconds": 0,
            "price": "100",
            "volume": "1000",
            "notional": "10",
            "tickRule": 1,
            "ticks": 1,
            "high": "100",
            "low": "100",
            "totalBuyVolume": "1000",
            "totalVolume": "1000",
            "totalBuyNotional": "10",
            "totalNotional": "10",
            "totalBuyTicks": 1,
            "totalTicks": 1,
        }

    def test_for_symbol_filters_significant_trade_filter(self):
        symbol = Symbol.objects.create(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            significant_trade_filter=1000,
        )
        timestamp = datetime(2026, 5, 10, 10, tzinfo=UTC)
        expected = WebSocketData.objects.create(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            significant_trade_filter=1000,
            timestamp=timestamp,
            filtered_trades=[self.get_trade()],
        )
        WebSocketData.objects.create(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            significant_trade_filter=500,
            timestamp=timestamp,
            filtered_trades=[self.get_trade()],
        )

        rows = list(WebSocketData.objects.for_symbol(symbol))

        self.assertEqual(rows, [expected])

    def test_get_data_frames_parses_trade_payloads(self):
        symbol = Symbol.objects.create(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            significant_trade_filter=1000,
        )
        data = WebSocketData.objects.create(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            significant_trade_filter=1000,
            timestamp=datetime(2026, 5, 10, 10, tzinfo=UTC),
            filtered_trades=[self.get_trade("6291886372")],
        )
        data.refresh_from_db()

        _raw_trades, _aggregated_trades, filtered_trades = data.get_data_frames(symbol)

        self.assertEqual(data.filtered_trades[0]["uid"], "6291886372")
        self.assertEqual(filtered_trades.iloc[0].uid, "6291886372")
        self.assertEqual(
            filtered_trades.iloc[0].timestamp.to_pydatetime(),
            datetime(2026, 5, 10, 10, 0, 10, tzinfo=UTC),
        )
        self.assertEqual(filtered_trades.iloc[0].totalVolume, Decimal("1000"))

    def test_get_data_frames_preserves_equal_timestamp_order(self):
        symbol = Symbol.objects.create(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            significant_trade_filter=1000,
        )
        data = WebSocketData(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            significant_trade_filter=1000,
            timestamp=datetime(2026, 5, 10, 10, tzinfo=UTC),
            filtered_trades=[self.get_trade("z"), self.get_trade("a")],
        )

        _raw_trades, _aggregated_trades, filtered_trades = data.get_data_frames(symbol)

        self.assertEqual(list(filtered_trades.uid), ["z", "a"])
