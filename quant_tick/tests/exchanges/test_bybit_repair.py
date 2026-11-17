import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import httpx2
import pandas as pd
import time_machine
from django.test import TestCase, override_settings

from quant_tick.constants import (
    RETRY_INDETERMINATE,
    Exchange,
    FileData,
    Frequency,
    SymbolType,
)
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
        symbol_id = (
            "BYBITSPOT_SPOT_BTC_USDT"
            if exchange == Exchange.BYBIT
            else "BYBIT_PERP_BTC_USDT"
        )
        timestamp = (self.start + timedelta(minutes=minute, seconds=1)).isoformat()
        return {
            "symbol_id": symbol_id,
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
                symbol_id = (
                    "BYBITSPOT_SPOT_BTC_USDT"
                    if exchange == Exchange.BYBIT
                    else "BYBIT_PERP_BTC_USDT"
                )
                self.assertEqual(
                    self.rest.call_args.args[0],
                    f"/v1/trades/{symbol_id}/history",
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
        self.metadata.side_effect = coinapi_history_start
        for exchange, vendor_exchange, symbol_id in (
            (Exchange.BYBIT, "BYBITSPOT", "BYBITSPOT_SPOT_BTC_USDT"),
            (Exchange.BYBIT_LINEAR, "BYBIT", "BYBIT_PERP_BTC_USDT"),
        ):
            with self.subTest(exchange=exchange):
                symbol = self.make_symbol(exchange)
                self.rest.return_value = [self.vendor(i, exchange) for i in range(3)]
                with patch(
                    "quant_tick.exchanges.coinapi.api.get_coinapi_response",
                    return_value=[
                        {
                            "symbol_id": symbol_id,
                            "data_trade_start": self.start.isoformat(),
                        }
                    ],
                ) as metadata:
                    api(symbol, self.start, self.end)
                    self.assertEqual(
                        coinapi_history_start(symbol_id), pd.Timestamp(self.start)
                    )
                    metadata.assert_called_once_with(
                        f"/v1/symbols/{vendor_exchange}/active",
                        {"filter_symbol_id": symbol_id},
                    )
                self.rest.assert_called_once()
                self.assertEqual(
                    self.rest.call_args.args[0], f"/v1/trades/{symbol_id}/history"
                )
                self.assertEqual(
                    TradeData.objects.filter(symbol=symbol, ok=True).count(), 3
                )
                self.rest.reset_mock()

    def test_partial_retry_keeps_daily_row_and_files_when_fetch_fails_or_is_empty(self):
        self.mock(
            "quant_tick.exchanges.hyperliquid.controllers.native_hyperliquid_candles",
            return_value=self.candles,
        )
        for exchange in (Exchange.BYBIT_LINEAR, Exchange.HYPERLIQUID):
            for ok, retry in ((False, True), (None, RETRY_INDETERMINATE)):
                for fails in (True, False):
                    with self.subTest(exchange=exchange, ok=ok, fails=fails):
                        symbol = self.get_symbol(
                            exchange=exchange,
                            api_symbol="BTC"
                            if exchange == Exchange.HYPERLIQUID
                            else "BTCUSDT",
                            symbol_type=SymbolType.PERPETUAL,
                        )
                        day = self.start.replace(hour=0)
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
                        filename = original.raw_data.name
                        storage = original.raw_data.storage
                        with storage.open(filename) as stream:
                            content = stream.read()
                        self.rest.side_effect = (
                            RuntimeError("fetch unavailable") if fails else None
                        )
                        self.rest.return_value = []
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
                        self.assertTrue(
                            TradeData.objects.filter(pk=original.pk).exists()
                        )
                        self.assertTrue(storage.exists(filename))
                        with storage.open(filename) as stream:
                            self.assertEqual(stream.read(), content)

    def test_new_failed_archive_minute_allows_native_then_vendor_recovery(self):
        self.mock(
            "quant_tick.exchanges.bybit.controllers.get_archive_followup_start",
            return_value=self.start,
        )
        for exchange, archive in (
            (Exchange.BYBIT_LINEAR, self.native_files[0]),
            (Exchange.BYBIT, self.native_files[1]),
        ):
            for native_quantity, source in (("1", "bybit"), ("0.5", "coinapi")):
                with self.subTest(exchange=exchange, source=source):
                    symbol = self.make_symbol(exchange)
                    # An older failed minute should not be paid for without --retry.
                    old = TradeData.objects.create(
                        symbol=symbol, timestamp=self.start, frequency=1, ok=False
                    )
                    self.websocket(symbol, 1, native_quantity)
                    timestamp = self.start + timedelta(minutes=1)
                    if exchange == Exchange.BYBIT:
                        trades = [
                            {
                                "id": "archive",
                                "timestamp": str(int(timestamp.timestamp() * 1000)),
                                "price": "100",
                                "volume": "0.5",
                                "side": "buy",
                            }
                        ]
                    else:
                        trades = [
                            {
                                "trdMatchID": "archive",
                                "timestamp": str(int(timestamp.timestamp())),
                                "symbol": "BTCUSDT",
                                "price": "100",
                                "size": "0.5",
                                "side": "Buy",
                            }
                        ]
                    archive.return_value = iter([pd.DataFrame(trades)])
                    self.rest.return_value = [self.vendor(1, exchange)]
                    api(symbol, self.start, self.start + timedelta(minutes=2))
                    self.assertTrue(TradeData.objects.filter(pk=old.pk).exists())
                    row = TradeData.objects.get(symbol=symbol, timestamp=timestamp)
                    self.assertIs(row.ok, True)
                    self.assertEqual(row.json_data["data_sources"], [source])
                    self.assertEqual(
                        row.get_data_frame(FileData.RAW).notional.sum(), Decimal(1)
                    )
                    if source == "bybit":
                        self.metadata.assert_not_called()
                        self.files.assert_not_called()
                        self.rest.assert_not_called()
                    else:
                        self.files.assert_called_once_with(
                            timestamp, timestamp + timedelta(minutes=1)
                        )
                        self.rest.assert_called_once()
                    archive.return_value = None
                    self.metadata.reset_mock()
                    self.files.reset_mock()
                    self.rest.reset_mock()
                    WebSocketData.objects.filter(
                        exchange=exchange, api_symbol="BTCUSDT"
                    ).delete()

    def archive_rows(self, exchange, minutes):
        rows = []
        for minute in minutes:
            timestamp = self.start + timedelta(minutes=minute)
            if exchange == Exchange.BYBIT:
                rows.append(
                    {
                        "id": str(minute),
                        "timestamp": str(int(timestamp.timestamp() * 1000)),
                        "price": "100",
                        "volume": "0.5",
                        "side": "buy",
                    }
                )
            else:
                rows.append(
                    {
                        "trdMatchID": str(minute),
                        "timestamp": str(int(timestamp.timestamp())),
                        "symbol": "BTCUSDT",
                        "price": "100",
                        "size": "0.5",
                        "side": "Buy",
                    }
                )
        return iter([pd.DataFrame(rows)])

    def test_repair_newest_archive_window_before_vendor_history_boundary(self):
        end = self.start + timedelta(minutes=62)
        self.native_candles.return_value = pd.DataFrame(
            {"notional": [Decimal(1)] * 62},
            index=pd.date_range(self.start, periods=62, freq="min"),
        )
        self.metadata.return_value = pd.Timestamp(self.start + timedelta(hours=1))
        for exchange, archive in (
            (Exchange.BYBIT_LINEAR, self.native_files[0]),
            (Exchange.BYBIT, self.native_files[1]),
        ):
            for retry in (False, True):
                with self.subTest(exchange=exchange, retry=retry):
                    symbol = self.make_symbol(exchange)
                    archive.return_value = self.archive_rows(exchange, [0, 60, 61])
                    self.rest.return_value = [
                        self.vendor(60, exchange),
                        self.vendor(61, exchange),
                    ]
                    api(symbol, self.start, end, retry=retry)
                    self.files.assert_called_once_with(
                        self.start + timedelta(hours=1), end
                    )
                    self.rest.assert_called_once()
                    params = self.rest.call_args.args[1]
                    self.assertEqual(
                        pd.Timestamp(params["time_start"]),
                        self.start + timedelta(hours=1),
                    )
                    self.assertEqual(pd.Timestamp(params["time_end"]), end)
                    rows = list(
                        TradeData.objects.filter(symbol=symbol).order_by("timestamp")
                    )
                    self.assertEqual([row.ok for row in rows], [False, True, True])
                    self.assertEqual(
                        [row.uid for row in rows], ["0", "vendor-60", "vendor-61"]
                    )
                    self.assertEqual(
                        rows[0].get_data_frame(FileData.RAW).notional.sum(),
                        Decimal("0.5"),
                    )
                    self.rest.reset_mock()
                    self.files.reset_mock()
                    archive.return_value = None

    @time_machine.travel(datetime(2026, 8, 1, 1, tzinfo=UTC), tick=False)
    def test_newer_missing_day_precedes_older_archive_repair(self):
        self.start = datetime(2026, 7, 31, 23, 59, tzinfo=UTC)
        end = self.start + timedelta(minutes=3)
        self.native_candles.return_value = pd.DataFrame(
            {"notional": [Decimal(1)] * 3},
            index=pd.date_range(self.start, periods=3, freq="min"),
        )
        self.metadata.return_value = pd.Timestamp(self.start + timedelta(minutes=1))
        for exchange, archive in (
            (Exchange.BYBIT_LINEAR, self.native_files[0]),
            (Exchange.BYBIT, self.native_files[1]),
        ):
            with self.subTest(exchange=exchange):
                symbol = self.make_symbol(exchange)
                archive.side_effect = lambda date, exchange=exchange: (
                    self.archive_rows(exchange, [0])
                    if date == self.start.date()
                    else None
                )
                self.rest.return_value = [
                    self.vendor(1, exchange),
                    self.vendor(2, exchange),
                ]
                api(symbol, self.start, end)
                self.files.assert_called_once_with(
                    self.start + timedelta(minutes=1), end
                )
                self.rest.assert_called_once()
                rows = list(
                    TradeData.objects.filter(symbol=symbol).order_by("timestamp")
                )
                self.assertEqual([row.ok for row in rows], [False, True, True])
                self.assertEqual(
                    [row.uid for row in rows], ["0", "vendor-1", "vendor-2"]
                )
                self.rest.reset_mock()
                self.files.reset_mock()
                archive.side_effect = None

    def test_overlapping_retry_windows_do_not_repeat_empty_vendor_request(self):
        for exchange, archive in (
            (Exchange.BYBIT_LINEAR, self.native_files[0]),
            (Exchange.BYBIT, self.native_files[1]),
        ):
            with self.subTest(exchange=exchange):
                symbol = self.make_symbol(exchange)
                archive.return_value = self.archive_rows(exchange, [0, 1, 2])
                api(symbol, self.start, self.end, retry=True)
                self.files.assert_called_once_with(self.start, self.end)
                self.rest.assert_called_once()
                self.assertEqual(
                    list(
                        TradeData.objects.filter(symbol=symbol).values_list(
                            "ok", flat=True
                        )
                    ),
                    [False] * 3,
                )
                self.rest.reset_mock()
                self.files.reset_mock()
                archive.return_value = None
