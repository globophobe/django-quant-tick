import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import httpx2
import pandas as pd
from django.test import TestCase, override_settings

from quant_tick.constants import Exchange, FileData, Frequency, SymbolType
from quant_tick.exchanges.api import api
from quant_tick.exchanges.coinapi.api import (
    _coinapi_history_start,
    coinapi_history_start,
)
from quant_tick.models import TradeData, WebSocketData

from ..base import BaseWriteTradeDataTest


@override_settings(
    STORAGES={"default": {"BACKEND": "django.core.files.storage.InMemoryStorage"}}
)
@patch.dict(os.environ, {"COINAPI_API_KEY": "test-key"})
class BybitRepairTest(BaseWriteTradeDataTest, TestCase):
    def setUp(self):
        super().setUp()
        self.start = datetime(2026, 7, 31, 1, tzinfo=UTC)
        self.end = self.start + timedelta(minutes=3)
        self.candles = pd.DataFrame(
            {"notional": [Decimal(1)] * 3, "volume": [Decimal(100)] * 3},
            index=pd.date_range(self.start, periods=3, freq="min"),
        )
        self.native_files = []
        for target in ("BybitTradesS3", "BybitSpotTradesS3"):
            self.native_files.append(
                self.mock(
                    f"quant_tick.exchanges.bybit.controllers.{target}.get_data_frame_chunks",
                    return_value=None,
                )
            )
        self.native_candles = self.mock(
            "quant_tick.exchanges.bybit.base.bybit_candles", return_value=self.candles
        )
        self.mock(
            "quant_tick.controllers.rest.ExchangeREST.get_websocket_timestamp_from",
            return_value=self.start,
        )
        self.metadata = self.mock(
            "quant_tick.exchanges.coinapi.repair.coinapi_history_start",
            return_value=pd.Timestamp(self.start),
        )
        self.files = self.mock(
            "quant_tick.exchanges.coinapi.flat_files.CoinAPITradeFiles.get_trades",
            return_value=None,
        )
        self.rest = self.mock(
            "quant_tick.exchanges.coinapi.controllers.get_coinapi_response",
            return_value=[],
        )
        self.vendor_candles = self.mock(
            "quant_tick.exchanges.coinapi.candles.get_coinapi_response"
        )

    def mock(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def make_symbol(self, exchange=Exchange.BYBIT_LINEAR):
        return self.get_symbol(
            exchange=exchange,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.SPOT
            if exchange == Exchange.BYBIT
            else SymbolType.PERPETUAL,
            save_aggregated=True,
            significant_trade_filter=1,
        )

    def vendor(self, minute, exchange=Exchange.BYBIT_LINEAR, size="1", side="BUY"):
        market = "SPOT" if exchange == Exchange.BYBIT else "PERP"
        timestamp = (self.start + timedelta(minutes=minute, seconds=1)).isoformat()
        return {
            "symbol_id": f"BYBIT_{market}_BTC_USDT",
            "uuid": f"vendor-{minute}",
            "id_trade": f"venue-{minute}",
            "time_exchange": timestamp,
            "time_coinapi": timestamp,
            "price": "100",
            "size": size,
            "taker_side": side,
        }

    def websocket(self, symbol, minute, quantity="1"):
        timestamp = self.start + timedelta(minutes=minute)
        WebSocketData.objects.create(
            exchange=symbol.exchange,
            api_symbol=symbol.api_symbol,
            significant_trade_filter=1,
            timestamp=timestamp,
            raw_trades=[
                {
                    "uid": f"native-{minute}",
                    "timestamp": timestamp.isoformat(),
                    "nanoseconds": 0,
                    "price": "100",
                    "notional": quantity,
                    "volume": str(Decimal(quantity) * 100),
                    "tickRule": 1,
                }
            ],
        )

    def test_native_then_files_then_rest_replace_whole_minutes_on_both_markets(self):
        for exchange in (Exchange.BYBIT, Exchange.BYBIT_LINEAR):
            with self.subTest(exchange=exchange):
                symbol = self.make_symbol(exchange)
                self.websocket(symbol, 0)
                self.websocket(symbol, 1, "0.5")
                self.files.return_value = [
                    {
                        **self.vendor(1, exchange, side="SELL"),
                        "coinapi_transport": "flat_files",
                    },
                    {
                        **self.vendor(2, exchange, size="0.5"),
                        "coinapi_transport": "flat_files",
                    },
                ]
                self.rest.return_value = [self.vendor(2, exchange)]
                api(symbol, self.start, self.end)
                self.files.assert_called_once_with(
                    self.start + timedelta(minutes=1), self.end
                )
                self.rest.assert_called_once()
                market = "SPOT" if exchange == Exchange.BYBIT else "PERP"
                self.assertEqual(
                    self.rest.call_args.args[0],
                    f"/v1/trades/BYBIT_{market}_BTC_USDT/history",
                )
                self.assertEqual(
                    pd.Timestamp(self.rest.call_args.args[1]["time_start"]),
                    self.start + timedelta(minutes=2),
                )
                rows = list(
                    TradeData.objects.filter(symbol=symbol).order_by("timestamp")
                )
                self.assertEqual([row.ok for row in rows], [True] * 3)
                self.assertEqual(
                    [row.uid for row in rows], ["native-0", "vendor-1", "vendor-2"]
                )
                self.assertEqual(
                    [row.json_data["data_sources"] for row in rows],
                    [["bybit"], ["coinapi"], ["coinapi"]],
                )
                for row in rows:
                    for field in (FileData.RAW, FileData.AGGREGATED, FileData.FILTERED):
                        frame = row.get_data_frame(field)
                        self.assertEqual(len(frame), 1)
                        self.assertEqual(frame.notional.sum(), Decimal(1))
                        self.assertEqual(frame.volume.sum(), Decimal(100))
                self.assertEqual(
                    rows[1].get_data_frame(FileData.RAW).tickRule.tolist(), [-1]
                )
                self.assertEqual(
                    rows[1].get_data_frame(FileData.RAW).id_trade.tolist(), ["venue-1"]
                )
                self.assertEqual(
                    rows[1].get_data_frame(FileData.RAW).coinapi_transport.tolist(),
                    ["flat_files"],
                )
                self.assertEqual(
                    rows[2].get_data_frame(FileData.RAW).coinapi_transport.tolist(),
                    ["rest"],
                )
                self.files.reset_mock()
                self.rest.reset_mock()
                self.metadata.reset_mock()
                api(symbol, self.start, self.end)
                with patch.dict(os.environ, {"COINAPI_API_KEY": ""}):
                    api(symbol, self.start, self.end)
                self.files.assert_not_called()
                self.rest.assert_not_called()
                self.metadata.assert_not_called()
                self.assertEqual(
                    list(
                        TradeData.objects.filter(symbol=symbol)
                        .order_by("timestamp")
                        .values_list("pk", flat=True)
                    ),
                    [row.pk for row in rows],
                )
        self.vendor_candles.assert_not_called()

    def test_complete_native_archive_never_calls_vendor(self):
        for exchange, archive in (
            (Exchange.BYBIT_LINEAR, self.native_files[0]),
            (Exchange.BYBIT, self.native_files[1]),
        ):
            with self.subTest(exchange=exchange):
                symbol = self.make_symbol(exchange)
                trades = []
                for minute in range(3):
                    timestamp = self.start + timedelta(minutes=minute)
                    if exchange == Exchange.BYBIT:
                        trades.append(
                            {
                                "id": str(minute),
                                "timestamp": str(int(timestamp.timestamp() * 1000)),
                                "price": "100",
                                "volume": "1",
                                "side": "buy",
                            }
                        )
                    else:
                        trades.append(
                            {
                                "trdMatchID": str(minute),
                                "timestamp": str(int(timestamp.timestamp())),
                                "symbol": "BTCUSDT",
                                "price": "100",
                                "size": "1",
                                "side": "Buy",
                            }
                        )
                archive.return_value = iter([pd.DataFrame(trades)])
                api(symbol, self.start, self.end)
                rows = TradeData.objects.filter(symbol=symbol)
                self.assertEqual(rows.count(), 3)
                self.assertTrue(all(row.ok for row in rows))
                archive.return_value = None
        self.metadata.assert_not_called()
        self.files.assert_not_called()
        self.rest.assert_not_called()
        self.vendor_candles.assert_not_called()

    def test_complete_websocket_and_zero_minutes_need_no_vendor(self):
        symbol = self.make_symbol()
        self.websocket(symbol, 0)
        self.websocket(symbol, 2)
        self.candles.loc[self.start + timedelta(minutes=1), ["notional", "volume"]] = (
            Decimal(0)
        )
        api(symbol, self.start, self.end)
        rows = list(TradeData.objects.filter(symbol=symbol).order_by("timestamp"))
        self.assertEqual([row.ok for row in rows], [True] * 3)
        self.assertEqual([row.uid for row in rows], ["native-0", "", "native-2"])
        self.metadata.assert_not_called()
        self.files.assert_not_called()
        self.rest.assert_not_called()

    def test_empty_or_failed_vendor_keeps_native_progress_and_gap_retryable(self):
        symbol = self.make_symbol()
        self.websocket(symbol, 0)
        api(symbol, self.start, self.end)
        native = TradeData.objects.get(symbol=symbol)
        self.assertTrue(native.ok)
        self.assertEqual(native.uid, "native-0")
        request = httpx2.Request("GET", "https://rest.coinapi.io")
        self.rest.side_effect = httpx2.HTTPStatusError(
            "credits exhausted",
            request=request,
            response=httpx2.Response(402, request=request),
        )
        with self.assertRaises(httpx2.HTTPStatusError):
            api(symbol, self.start, self.end)
        self.assertEqual(TradeData.objects.get(symbol=symbol).pk, native.pk)
        self.rest.side_effect = None
        self.rest.return_value = [self.vendor(1), self.vendor(2)]
        api(symbol, self.start, self.end)
        self.assertEqual(
            list(TradeData.objects.filter(symbol=symbol).values_list("ok", flat=True)),
            [True] * 3,
        )

    def test_retry_uses_recovered_websocket_before_paying_and_preserves_good_minutes(
        self,
    ):
        symbol = self.make_symbol()
        old = TradeData.objects.create(
            symbol=symbol, timestamp=self.start, frequency=1, ok=True
        )
        TradeData.objects.create(
            symbol=symbol,
            timestamp=self.start + timedelta(minutes=1),
            frequency=1,
            ok=False,
        )
        self.websocket(symbol, 1)
        self.rest.return_value = [self.vendor(2)]
        api(symbol, self.start, self.end, retry=True)
        self.assertTrue(TradeData.objects.filter(pk=old.pk).exists())
        rows = list(TradeData.objects.filter(symbol=symbol).order_by("timestamp"))
        self.assertEqual([row.ok for row in rows], [True] * 3)
        self.assertEqual(rows[1].uid, "native-1")
        self.files.assert_called_once_with(self.start + timedelta(minutes=2), self.end)

    def test_provider_quantity_mismatch_remains_failed_until_retry(self):
        symbol = self.make_symbol()
        self.rest.return_value = [self.vendor(0, size="0.5")]
        end = self.start + timedelta(minutes=1)
        api(symbol, self.start, end)
        self.assertIs(TradeData.objects.get(symbol=symbol).ok, False)
        self.rest.return_value = [self.vendor(0)]
        api(symbol, self.start, end, retry=True)
        self.assertIs(TradeData.objects.get(symbol=symbol).ok, True)

    def test_history_boundary_avoids_unavailable_paid_history(self):
        symbol = self.make_symbol()
        self.metadata.return_value = pd.Timestamp(
            self.start + timedelta(minutes=1, seconds=10)
        )
        self.rest.return_value = [self.vendor(2)]
        api(symbol, self.start, self.end)
        self.files.assert_called_once_with(self.start + timedelta(minutes=2), self.end)
        row = TradeData.objects.get(symbol=symbol)
        self.assertTrue(row.ok)
        self.assertEqual(row.uid, "vendor-2")

    def test_missing_metadata_keeps_gap_without_trade_requests(self):
        symbol = self.make_symbol()
        self.metadata.return_value = None
        api(symbol, self.start, self.end)
        self.assertFalse(TradeData.objects.filter(symbol=symbol).exists())
        self.files.assert_not_called()
        self.rest.assert_not_called()

    def test_partial_missing_hour_boundaries_are_preserved(self):
        self.candles = pd.DataFrame(
            {"notional": [Decimal(1)] * 120},
            index=pd.date_range(self.start, periods=120, freq="min"),
        )
        self.native_candles.return_value = self.candles
        symbol = self.make_symbol()
        existing = TradeData.objects.bulk_create(
            [
                TradeData(
                    symbol=symbol,
                    timestamp=self.start + timedelta(minutes=i),
                    frequency=1,
                    ok=True,
                )
                for i in range(120)
                if not 15 <= i < 75
            ]
        )

        def files(start, end):
            return [
                self.vendor(i)
                for i in range(120)
                if start <= self.start + timedelta(minutes=i) < end
            ]

        self.files.side_effect = files
        api(symbol, self.start, self.start + timedelta(hours=2))
        self.assertEqual(
            [call.args for call in self.files.call_args_list],
            [
                (
                    self.start + timedelta(minutes=60),
                    self.start + timedelta(minutes=75),
                ),
                (
                    self.start + timedelta(minutes=15),
                    self.start + timedelta(minutes=60),
                ),
            ],
        )
        self.assertEqual(TradeData.objects.filter(symbol=symbol, ok=True).count(), 120)
        self.assertEqual(
            TradeData.objects.filter(pk__in=[row.pk for row in existing]).count(),
            len(existing),
        )
        self.rest.assert_not_called()

    def test_partial_request_preserves_overlapping_complete_hour(self):
        symbol = self.make_symbol()
        old = TradeData.objects.create(
            symbol=symbol, timestamp=self.start, frequency=Frequency.HOUR, ok=True
        )
        self.rest.return_value = [self.vendor(1)]
        api(
            symbol, self.start + timedelta(minutes=1), self.start + timedelta(minutes=2)
        )
        self.assertTrue(TradeData.objects.filter(pk=old.pk).exists())
        self.metadata.assert_not_called()
        self.files.assert_not_called()
        self.rest.assert_not_called()

    def test_metadata_uses_exact_bybit_market_and_cache(self):
        _coinapi_history_start.cache_clear()
        self.addCleanup(_coinapi_history_start.cache_clear)
        with patch(
            "quant_tick.exchanges.coinapi.api.get_coinapi_response",
            return_value=[
                {
                    "symbol_id": "BYBIT_PERP_BTC_USDT",
                    "data_trade_start": self.start.isoformat(),
                }
            ],
        ) as metadata:
            self.assertEqual(
                coinapi_history_start("BYBIT_PERP_BTC_USDT"), pd.Timestamp(self.start)
            )
            coinapi_history_start("BYBIT_PERP_BTC_USDT")
            metadata.assert_called_once_with(
                "/v1/symbols/BYBIT/active", {"filter_symbol_id": "BYBIT_PERP_BTC_USDT"}
            )
            self.assertIsNone(coinapi_history_start("BYBIT_SPOT_BTC_USDT"))
