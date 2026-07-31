from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import pandas as pd
from django.test import SimpleTestCase, TestCase

from quant_tick.constants import Exchange, FileData, SymbolType
from quant_tick.exchanges.api import api
from quant_tick.exchanges.hyperliquid.controllers import (
    HyperliquidTradesWebSocket,
    hyperliquid_trades,
)
from quant_tick.models import TradeData, WebSocketData

from ..base import BaseWriteTradeDataTest


class HyperliquidTradesTest(SimpleTestCase):
    def test_trades_use_websocket_controller(self):
        symbol = Mock()
        timestamp_from = datetime(2026, 7, 31, 1, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(minutes=1)
        on_data_frame = Mock()

        with patch(
            "quant_tick.exchanges.hyperliquid.controllers.HyperliquidTradesWebSocket"
        ) as websocket:
            hyperliquid_trades(
                symbol,
                timestamp_from,
                timestamp_to,
                on_data_frame,
            )

        websocket.assert_called_once_with(
            symbol,
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            on_data_frame=on_data_frame,
            retry=False,
            verbose=False,
        )
        websocket.return_value.main.assert_called_once_with()

    def test_websocket_controller_uses_one_minute_candles(self):
        symbol = Mock(api_symbol="BTC")
        timestamp_from = datetime(2026, 7, 31, 1, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(minutes=1)
        expected = pd.DataFrame([])
        controller = HyperliquidTradesWebSocket(
            symbol,
            timestamp_from,
            timestamp_to,
            Mock(),
        )

        with patch(
            "quant_tick.exchanges.hyperliquid.controllers.hyperliquid_candles",
            return_value=expected,
        ) as candles:
            result = controller.get_candles(timestamp_from, timestamp_to)

        candles.assert_called_once_with(
            "BTC",
            timestamp_from,
            timestamp_to,
            resolution="1m",
        )
        self.assertIs(result, expected)


class HyperliquidWebSocketValidationTest(BaseWriteTradeDataTest, TestCase):
    def setUp(self):
        super().setUp()
        self.timestamp_from = datetime(2026, 7, 31, 1, tzinfo=UTC)

    @staticmethod
    def get_filtered_trade(uid: str, timestamp: datetime, notional: str) -> dict:
        price = Decimal("64000")
        base_quantity = Decimal(notional)
        quote_quantity = price * base_quantity
        return {
            "uid": uid,
            "timestamp": timestamp.isoformat(),
            "nanoseconds": 0,
            "price": str(price),
            "volume": str(quote_quantity),
            "notional": str(base_quantity),
            "tickRule": 1,
            "ticks": 1,
            "high": str(price),
            "low": str(price),
            "totalBuyVolume": str(quote_quantity),
            "totalVolume": str(quote_quantity),
            "totalBuyNotional": str(base_quantity),
            "totalNotional": str(base_quantity),
            "totalBuyTicks": 1,
            "totalTicks": 1,
        }

    def create_websocket_minute(
        self,
        timestamp: datetime,
        uid: str,
        notional: str,
    ) -> None:
        WebSocketData.objects.create(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="BTC",
            significant_trade_filter=1000,
            timestamp=timestamp,
            filtered_trades=[self.get_filtered_trade(uid, timestamp, notional)],
        )

    def test_websocket_minutes_retain_validation_state(self):
        second_minute = self.timestamp_from + timedelta(minutes=1)
        third_minute = self.timestamp_from + timedelta(minutes=2)
        timestamp_to = self.timestamp_from + timedelta(minutes=3)
        symbol = self.get_symbol(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
            save_raw=False,
            significant_trade_filter=1000,
        )
        self.create_websocket_minute(self.timestamp_from, "valid", "1.5")
        self.create_websocket_minute(second_minute, "invalid", "2")
        self.create_websocket_minute(third_minute, "indeterminate", "3")
        candles = pd.DataFrame(
            {"notional": [Decimal("1.5"), Decimal("1.5")]},
            index=[self.timestamp_from, second_minute],
        )

        with (
            patch(
                "quant_tick.exchanges.hyperliquid.controllers.hyperliquid_candles",
                return_value=candles,
            ),
            patch(
                "quant_tick.controllers.rest.ExchangeWebSocket."
                "get_websocket_timestamp_from",
                return_value=self.timestamp_from,
            ),
        ):
            api(symbol, self.timestamp_from, timestamp_to)

        rows = list(TradeData.objects.filter(symbol=symbol).order_by("timestamp"))
        self.assertEqual([row.ok for row in rows], [True, False, None])
        self.assertEqual(
            [row.uid for row in rows],
            ["valid", "invalid", "indeterminate"],
        )
        for row in rows:
            self.assertTrue(row.has_data_frame(FileData.FILTERED))
