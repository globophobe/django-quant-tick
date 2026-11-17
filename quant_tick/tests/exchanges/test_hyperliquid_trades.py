import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import httpx2
import pandas as pd
import time_machine
from django.test import SimpleTestCase, TestCase, override_settings

from quant_tick.constants import RETRY_INDETERMINATE, Exchange, FileData, SymbolType
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
            allow_quicknode=False,
        )
        self.assertIs(result, expected)


@patch.dict(os.environ, {"QUICKNODE_API_KEY": ""})
class HyperliquidWebSocketValidationTest(BaseWriteTradeDataTest, TestCase):
    def setUp(self):
        super().setUp()
        self.timestamp_from = datetime(2026, 7, 31, 1, tzinfo=UTC)

    @staticmethod
    def get_filtered_trade(uid: str, timestamp: datetime, notional: str) -> dict:
        price = Decimal(64000)
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
            self.assertEqual(row.json_data["data_sources"], ["hyperliquid"])


@override_settings(
    STORAGES={"default": {"BACKEND": "django.core.files.storage.InMemoryStorage"}}
)
@patch.dict(os.environ, {"QUICKNODE_API_KEY": "test-key"})
class HyperliquidBackfillTest(BaseWriteTradeDataTest, TestCase):
    def setUp(self):
        super().setUp()
        self.start = datetime(2026, 7, 31, 1, tzinfo=UTC)
        self.end = self.start + timedelta(minutes=3)
        self.symbol = self.get_symbol(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
            save_aggregated=True,
            significant_trade_filter=1,
        )
        self.candles = pd.DataFrame(
            {"notional": [Decimal(1)] * 3},
            index=pd.date_range(self.start, periods=3, freq="min"),
        )
        for target, value in (
            (
                "quant_tick.exchanges.hyperliquid.controllers.native_hyperliquid_candles",
                self.candles,
            ),
            (
                "quant_tick.exchanges.hyperliquid.controllers.history_start",
                pd.Timestamp(self.start) - pd.Timedelta(days=1),
            ),
        ):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def websocket_minute(self, minute, quantity="1"):
        timestamp = self.start + timedelta(minutes=minute)
        WebSocketData.objects.create(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="BTC",
            significant_trade_filter=1,
            timestamp=timestamp,
            raw_trades=[
                {
                    "uid": "native-" + str(minute),
                    "timestamp": timestamp.isoformat(),
                    "nanoseconds": 0,
                    "price": "100",
                    "notional": quantity,
                    "volume": str(Decimal(quantity) * 100),
                    "tickRule": 1,
                }
            ],
        )

    def vendor_minute(self, minute):
        return {
            "block_number": minute + 100,
            "trade_id": 123456789 + minute,
            "timestamp": pd.Timestamp(
                self.start + timedelta(minutes=minute, seconds=1)
            ),
            "price": Decimal(100),
            "size": Decimal(1),
            "side": "B",
        }

    def test_missing_or_invalid_native_minute_uses_only_vendor_remainder(self):
        self.websocket_minute(1, "0.5")
        self.websocket_minute(2)

        def arriving_websocket(*args, **kwargs):
            self.websocket_minute(0)
            return self.candles

        with (
            time_machine.travel(self.end, tick=False),
            patch(
                "quant_tick.exchanges.hyperliquid.controllers.native_hyperliquid_candles",
                side_effect=arriving_websocket,
            ),
            patch(
                "quant_tick.exchanges.quicknode.trades.iter_trades",
                return_value=[self.vendor_minute(1)],
            ) as request,
            patch(
                "quant_tick.exchanges.hyperliquid.candles.quicknode_candles"
            ) as vendor_candles,
        ):
            api(self.symbol, self.start, self.end)
            request.assert_called_once()
            request.assert_called_once_with(
                "BTC",
                self.start + timedelta(minutes=1),
                self.start + timedelta(minutes=2),
            )
            vendor_candles.assert_not_called()
            request.reset_mock()
            api(self.symbol, self.start, self.end)
            request.assert_not_called()
            with (
                patch.dict(os.environ, {"QUICKNODE_API_KEY": ""}),
                patch(
                    "quant_tick.exchanges.hyperliquid.controllers.hyperliquid_candles",
                    return_value=self.candles,
                ),
            ):
                api(self.symbol, self.start, self.end)
            request.assert_not_called()
        rows = list(TradeData.objects.filter(symbol=self.symbol).order_by("timestamp"))
        self.assertEqual([row.ok for row in rows], [True] * 3)
        self.assertEqual(
            [row.json_data["data_sources"] for row in rows],
            [["hyperliquid"], ["quicknode"], ["hyperliquid"]],
        )
        for row in rows:
            for field in (FileData.RAW, FileData.AGGREGATED, FileData.FILTERED):
                frame = row.get_data_frame(field)
                self.assertEqual(len(frame), 1)
                self.assertEqual(frame.notional.sum(), Decimal(1))
                self.assertEqual(
                    frame.data_source.tolist(), row.json_data["data_sources"]
                )

    def test_retry_prefers_recovered_native_minute_without_paid_request(self):
        self.websocket_minute(0)
        TradeData.objects.create(
            symbol=self.symbol, timestamp=self.start, frequency=1, ok=False
        )
        with (
            time_machine.travel(self.start + timedelta(minutes=59), tick=False),
            patch(
                "quant_tick.exchanges.quicknode.trades.iter_trades",
                return_value=[],
            ) as request,
            patch(
                "quant_tick.exchanges.hyperliquid.controllers.history_start",
                return_value=pd.Timestamp(self.start),
            ) as metadata,
        ):
            api(self.symbol, self.start, self.start + timedelta(minutes=1), retry=True)
        request.assert_not_called()
        metadata.assert_not_called()
        stored = TradeData.objects.get(symbol=self.symbol)
        self.assertTrue(stored.ok)
        self.assertEqual(stored.json_data["data_sources"], ["hyperliquid"])

    def test_native_minute_between_gaps_is_not_replaced_by_vendor(self):
        self.websocket_minute(1)
        with patch(
            "quant_tick.exchanges.quicknode.trades.iter_trades",
            side_effect=[[self.vendor_minute(2)], [self.vendor_minute(0)]],
        ) as request:
            api(self.symbol, self.start, self.end)
        self.assertEqual(request.call_count, 2)
        windows = [
            (
                pd.Timestamp(call.args[1]),
                pd.Timestamp(call.args[2]),
            )
            for call in request.call_args_list
        ]
        self.assertEqual(
            windows,
            [
                (self.start + timedelta(minutes=2), self.end),
                (self.start, self.start + timedelta(minutes=1)),
            ],
        )
        rows = list(TradeData.objects.filter(symbol=self.symbol).order_by("timestamp"))
        self.assertEqual(
            [row.json_data["data_sources"] for row in rows],
            [["quicknode"], ["hyperliquid"], ["quicknode"]],
        )

    def test_empty_vendor_response_leaves_missing_history_available_for_next_pass(self):
        with patch(
            "quant_tick.exchanges.quicknode.trades.iter_trades",
            return_value=[],
        ) as request:
            api(self.symbol, self.start, self.end)
            self.assertFalse(TradeData.objects.filter(symbol=self.symbol).exists())
            api(self.symbol, self.start, self.end)
        self.assertEqual(request.call_count, 2)

    def test_provider_failure_preserves_native_progress_and_leaves_gap_retryable(self):
        self.websocket_minute(0)
        request = httpx2.Request("POST", "https://api.quicknode.com/sql/rest/v1/query")
        response = httpx2.Response(402, request=request)
        error = httpx2.HTTPStatusError(
            "credits exhausted", request=request, response=response
        )
        with (
            patch(
                "quant_tick.exchanges.quicknode.trades.iter_trades",
                side_effect=error,
            ),
            self.assertRaises(httpx2.HTTPStatusError),
        ):
            api(self.symbol, self.start, self.end)
        stored = TradeData.objects.get(symbol=self.symbol)
        self.assertEqual(stored.timestamp, self.start)
        self.assertTrue(stored.ok)

    def test_history_boundary_stops_without_empty_success_or_trade_requests(self):
        with (
            patch(
                "quant_tick.exchanges.hyperliquid.controllers.history_start",
                return_value=pd.Timestamp(self.end),
            ),
            patch("quant_tick.exchanges.quicknode.trades.iter_trades") as request,
        ):
            api(self.symbol, self.start, self.end)
        request.assert_not_called()
        self.assertFalse(TradeData.objects.filter(symbol=self.symbol).exists())

    def test_history_boundary_skips_older_gap_but_still_fills_later_gap(self):
        self.websocket_minute(1)
        with (
            patch(
                "quant_tick.exchanges.hyperliquid.controllers.history_start",
                return_value=pd.Timestamp(self.start) + pd.Timedelta(minutes=1),
            ),
            patch(
                "quant_tick.exchanges.quicknode.trades.iter_trades",
                return_value=[self.vendor_minute(2)],
            ) as request,
        ):
            api(self.symbol, self.start, self.end)
        request.assert_called_once()
        self.assertEqual(
            list(
                TradeData.objects.filter(symbol=self.symbol)
                .order_by("timestamp")
                .values_list("timestamp", flat=True)
            ),
            [self.start + timedelta(minutes=1), self.start + timedelta(minutes=2)],
        )

    def test_partial_retry_keeps_daily_data_when_provider_fails_or_is_empty(self):
        day = self.start.replace(hour=0)
        for ok, retry in ((False, True), (None, RETRY_INDETERMINATE)):
            for fails in (True, False):
                with self.subTest(ok=ok, fails=fails):
                    symbol = self.get_symbol(
                        exchange=Exchange.HYPERLIQUID,
                        api_symbol="BTC",
                        symbol_type=SymbolType.PERPETUAL,
                    )
                    original = TradeData.write(
                        symbol,
                        day,
                        day + timedelta(days=1),
                        pd.DataFrame([]),
                        raw_trades=self.get_raw(
                            day, price=Decimal(100), notional=Decimal(1)
                        ),
                    )[0]
                    original.ok = ok
                    original.save(update_fields=["ok"])
                    filename, storage = (
                        original.raw_data.name,
                        original.raw_data.storage,
                    )
                    with storage.open(filename) as stream:
                        content = stream.read()
                    with patch(
                        "quant_tick.exchanges.quicknode.trades.iter_trades",
                        side_effect=RuntimeError("fetch unavailable")
                        if fails
                        else None,
                        return_value=[],
                    ):
                        if fails:
                            with self.assertRaisesRegex(
                                RuntimeError, "fetch unavailable"
                            ):
                                api(
                                    symbol,
                                    self.start,
                                    self.start + timedelta(minutes=1),
                                    retry=retry,
                                )
                        else:
                            api(
                                symbol,
                                self.start,
                                self.start + timedelta(minutes=1),
                                retry=retry,
                            )
                    self.assertTrue(TradeData.objects.filter(pk=original.pk).exists())
                    with storage.open(filename) as stream:
                        self.assertEqual(stream.read(), content)

    def test_both_partial_hour_boundaries_remain_eligible_for_repair(self):
        start = self.start + timedelta(minutes=15)
        end = start + timedelta(hours=1)
        with (
            patch(
                "quant_tick.exchanges.hyperliquid.controllers.native_hyperliquid_candles",
                return_value=pd.DataFrame(
                    {"notional": [Decimal(1)] * 60},
                    index=pd.date_range(start, periods=60, freq="min"),
                ),
            ),
            patch(
                "quant_tick.exchanges.quicknode.trades.iter_trades", return_value=[]
            ) as fetch,
        ):
            api(self.symbol, start, end)
        boundary = pd.Timestamp(start).ceil("h")
        self.assertEqual(
            [(c.args[1], c.args[2]) for c in fetch.call_args_list],
            [(boundary, end), (start, boundary)],
        )
