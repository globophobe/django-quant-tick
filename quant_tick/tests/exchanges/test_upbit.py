from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import time_machine
from django.test import SimpleTestCase, TestCase, override_settings

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import api, candles_api, exchange_candles, trades_api
from quant_tick.exchanges.upbit.candles import upbit_candles
from quant_tick.exchanges.upbit.controllers import UpbitArchive, UpbitTrades
from quant_tick.lib.download import ArchiveDownloadError
from quant_tick.models import ExchangeCandleData, TradeData

from ..base import BaseSymbolTest

START = datetime(2026, 10, 10, tzinfo=UTC)
MS = int(START.timestamp() * 1000)


def trade(trade_id, timestamp=MS, price="100000000", side="BID"):
    return {
        "market": "KRW-BTC",
        "sequential_id": trade_id,
        "timestamp": timestamp,
        "trade_price": Decimal(price),
        "trade_volume": Decimal("0.01"),
        "ask_bid": side,
    }


class UpbitMarketTest(SimpleTestCase):
    def controller(self):
        return UpbitTrades(
            SimpleNamespace(api_symbol="KRW-BTC"),
            START,
            START + timedelta(minutes=1),
            Mock(),
        )

    def test_cursor_preserves_second_ties_and_quote_currency(self):
        controller = self.controller()
        with (
            patch(
                "quant_tick.exchanges.upbit.controllers.get_current_time",
                return_value=START + timedelta(days=2),
            ),
            patch(
                "quant_tick.exchanges.upbit.controllers.get_upbit_response",
                side_effect=[
                    [trade(3), trade(2, side="ASK")],
                    [trade(1), trade(0, MS - 1)],
                ],
            ) as request,
        ):
            rows, _, _ = controller.iter_api(START, START + timedelta(minutes=1))
        frame = controller.get_data_frame(controller.parse_data(rows))
        self.assertEqual(frame.uid.tolist(), ["1", "2", "3"])
        self.assertEqual(frame.tickRule.tolist(), [1, -1, 1])
        self.assertEqual(frame.volume.sum(), Decimal(3000000))
        self.assertEqual(frame.notional.sum(), Decimal("0.03"))
        self.assertEqual(request.call_args_list[0].args[1]["days_ago"], 2)
        self.assertEqual(request.call_args_list[1].args[1]["cursor"], 2)

    def test_midnight_requests_previous_utc_day_without_zero_time_bound(self):
        with (
            patch(
                "quant_tick.exchanges.upbit.controllers.get_current_time",
                return_value=START + timedelta(days=2),
            ),
            patch(
                "quant_tick.exchanges.upbit.controllers.get_upbit_response",
                return_value=[],
            ) as request,
        ):
            self.controller().iter_api(START - timedelta(minutes=1), START)
        self.assertEqual(request.call_args.args[1]["days_ago"], 3)
        self.assertNotIn("to", request.call_args.args[1])

    def test_retention_boundary_and_repeated_pages_fail_explicitly(self):
        with (
            patch(
                "quant_tick.exchanges.upbit.controllers.get_current_time",
                return_value=START + timedelta(days=8),
            ),
            self.assertRaisesRegex(ValueError, "seven days"),
        ):
            self.controller().iter_api(START, START + timedelta(minutes=1))
        with (
            patch(
                "quant_tick.exchanges.upbit.controllers.get_current_time",
                return_value=START + timedelta(days=2),
            ),
            patch(
                "quant_tick.exchanges.upbit.controllers.get_upbit_response",
                side_effect=[[trade(1)], [trade(1)]],
            ),
            self.assertRaisesRegex(ValueError, "advance"),
        ):
            self.controller().iter_api(START, START + timedelta(minutes=1))

    def test_candles_use_bucket_time_and_preserve_precise_krw_turnover(self):
        row = {
            "candle_date_time_utc": "2026-10-10T00:00:00",
            "timestamp": MS + 59000,
            "opening_price": Decimal(100000000),
            "high_price": Decimal(100000001),
            "low_price": Decimal(99999999),
            "trade_price": Decimal(100000001),
            "candle_acc_trade_volume": Decimal("0.12345678"),
            "candle_acc_trade_price": Decimal("12345678.12345678"),
        }
        with patch(
            "quant_tick.exchanges.upbit.candles.get_upbit_response", return_value=[row]
        ):
            frame = upbit_candles("KRW-BTC", START, START + timedelta(minutes=1))
        self.assertEqual(list(frame.index), [START])
        self.assertEqual(frame.iloc[0].volume, Decimal("12345678.12345678"))
        self.assertEqual(frame.iloc[0].notional, Decimal("0.12345678"))

    def test_archive_ids_are_scoped_to_utc_day(self):
        frame = pd.DataFrame(
            [
                ["0", str(MS), "1", "100", "BID"],
                ["0", str(MS + 86400000), "1", "100", "ASK"],
            ],
            columns=UpbitArchive.csv_columns,
        )
        controller = UpbitArchive(
            SimpleNamespace(api_symbol="KRW-BTC"),
            START,
            START + timedelta(days=2),
            Mock(),
        )
        result = controller.parse_dtypes_and_strip_columns(frame)
        self.assertEqual(result.uid.tolist(), ["20261010:0", "20261011:0"])


class UpbitCandleCollectionTest(BaseSymbolTest, TestCase):
    def test_eight_hour_collection_preserves_traded_windows_with_quiet_intervals(self):
        start = datetime(2026, 10, 11, tzinfo=UTC)
        cases = (
            (
                "BTC-BAT",
                (12, 16),
                [(8, 13, 23, 12, 14, 10, 100), (16, 17, 27, 16, 18, 10, 100)],
            ),
            (
                "KRW-BTC",
                (0, 4, 8, 12, 16, 20),
                [
                    (0, 1, 15, 0, 6, 20, 200),
                    (8, 9, 23, 8, 14, 20, 200),
                    (16, 17, 31, 16, 22, 20, 200),
                ],
            ),
        )
        for market, hours, expected in cases:
            with self.subTest(market=market):
                symbol = self.get_symbol(
                    exchange=Exchange.UPBIT,
                    api_symbol=market,
                    exchange_candle_resolution="8h",
                )
                page = [
                    {
                        "candle_date_time_utc": (
                            start + timedelta(hours=hour)
                        ).strftime("%Y-%m-%dT%H:%M:%S"),
                        "opening_price": Decimal(hour + 1),
                        "high_price": Decimal(hour + 11),
                        "low_price": Decimal(hour),
                        "trade_price": Decimal(hour + 2),
                        "candle_acc_trade_volume": Decimal(10),
                        "candle_acc_trade_price": Decimal(100),
                    }
                    for hour in reversed(hours)
                ]
                with patch(
                    "quant_tick.exchanges.upbit.candles.get_upbit_response",
                    side_effect=[page, []],
                ):
                    exchange_candles(symbol, start, start + timedelta(days=1))
                rows = list(
                    ExchangeCandleData.objects.filter(symbol=symbol, frequency=480)
                )
                self.assertEqual(len(rows), len(expected))
                for row, (hour, *values) in zip(rows, expected, strict=True):
                    self.assertEqual(row.timestamp, start + timedelta(hours=hour))
                    self.assertEqual(
                        [
                            row.open,
                            row.high,
                            row.low,
                            row.close,
                            row.notional,
                            row.volume,
                        ],
                        [Decimal(value) for value in values],
                    )


@override_settings(
    STORAGES={"default": {"BACKEND": "django.core.files.storage.InMemoryStorage"}}
)
class UpbitArchiveCollectionTest(BaseSymbolTest, TestCase):
    def test_archive_orders_across_chunks_and_validates_stored_hours(self):
        start = datetime(2022, 5, 1, tzinfo=UTC)
        ms = int(start.timestamp() * 1000)
        symbol = self.get_symbol(exchange=Exchange.UPBIT, api_symbol="KRW-BTC")
        chunk = pd.DataFrame(
            [
                list(UpbitArchive.csv_columns),
                ["0", str(ms), "0.01", "100000000", "BID"],
                ["1", str(ms + 3600000), "0.02", "100000001", "ASK"],
                ["2", str(ms), "0.03", "100000002", "BID"],
            ],
            columns=UpbitArchive.csv_columns,
        )
        candles = pd.DataFrame(
            {"notional": [Decimal("0.04"), Decimal("0.02")]},
            index=pd.date_range(start, periods=2, freq="h"),
        )
        candles.index.name = "timestamp"
        with (
            patch(
                "quant_tick.exchanges.upbit.controllers.zip_downloader",
                return_value=chunk,
            ),
            patch.object(UpbitArchive, "archive_chunksize", 2),
            patch(
                "quant_tick.exchanges.upbit.controllers.upbit_candles",
                return_value=candles,
            ),
        ):
            api(symbol, start, start + timedelta(days=1))
        rows = TradeData.objects.filter(symbol=symbol)
        self.assertEqual(rows.count(), 24)
        self.assertFalse(rows.exclude(ok=True).exists())
        first = rows.get(timestamp=start)
        self.assertEqual(first.uid, "20220501:0")
        frame = first.get_data_frame("raw_data")
        self.assertEqual(
            frame.price.tolist(),
            [Decimal(100000000), Decimal(100000002)],
        )
        self.assertEqual(first.json_data["candle"]["close"], Decimal(100000002))
        self.assertEqual(
            rows.get(timestamp=start + timedelta(hours=1)).uid, "20220501:1"
        )

    def test_unpublished_archive_falls_back_to_rest_and_validates_stored_trades(self):
        for days_ago in (0, 1, 7):
            with self.subTest(days_ago=days_ago):
                symbol = self.get_symbol(exchange=Exchange.UPBIT, api_symbol="KRW-BTC")
                candles = pd.DataFrame(
                    {"notional": [Decimal("0.02")]},
                    index=pd.DatetimeIndex([START], name="timestamp"),
                )
                with (
                    time_machine.travel(
                        START + timedelta(days=days_ago, hours=1), tick=False
                    ),
                    patch(
                        "quant_tick.lib.download.httpx2.get",
                        return_value=Mock(status_code=403),
                    ) as archive,
                    patch(
                        "quant_tick.exchanges.upbit.controllers.get_upbit_response",
                        return_value=[trade(2, side="ASK"), trade(1), trade(0, MS - 1)],
                    ) as rest,
                    patch(
                        "quant_tick.exchanges.upbit.controllers.upbit_candles",
                        return_value=candles,
                    ),
                ):
                    api(symbol, START, START + timedelta(minutes=1))
                stored = TradeData.objects.get(symbol=symbol)
                self.assertIs(stored.ok, True)
                frame = stored.get_data_frame("raw_data")
                self.assertEqual(stored.uid, "1")
                self.assertEqual(len(frame), 2)
                self.assertEqual(frame.tickRule.tolist(), [1, -1])
                self.assertEqual(frame.notional.sum(), Decimal("0.02"))
                self.assertEqual(frame.volume.sum(), Decimal(2000000))
                self.assertEqual(rest.call_args.args[1]["days_ago"], days_ago)
                self.assertEqual(
                    archive.call_args.args[0],
                    "https://crix-data.upbit.com/trade/KRW-BTC/daily/2026/KRW-BTC_trade_20261010.zip",
                )

    def test_old_forbidden_archive_and_recent_server_error_remain_failures(self):
        symbol = self.get_symbol(exchange=Exchange.UPBIT, api_symbol="KRW-BTC")
        for days_ago, status in ((8, 403), (0, 503)):
            with (
                self.subTest(days_ago=days_ago, status=status),
                time_machine.travel(
                    START + timedelta(days=days_ago, hours=1), tick=False
                ),
                patch(
                    "quant_tick.lib.download.httpx2.get",
                    return_value=Mock(status_code=status),
                ),
                self.assertRaisesRegex(ArchiveDownloadError, f"HTTP {status}"),
            ):
                api(symbol, START, START + timedelta(minutes=1))

    def test_completed_archive_partitions_skip_recent_rest_followup(self):
        symbol = self.get_symbol(exchange=Exchange.UPBIT, api_symbol="KRW-BTC")
        TradeData.objects.create(symbol=symbol, timestamp=START, frequency=60, ok=True)
        with (
            patch(
                "quant_tick.exchanges.upbit.controllers.get_current_time",
                return_value=START + timedelta(days=1),
            ),
            patch(
                "quant_tick.exchanges.upbit.controllers.zip_downloader",
                return_value=None,
            ),
            patch(
                "quant_tick.exchanges.upbit.controllers.get_upbit_response"
            ) as request,
        ):
            api(symbol, START, START + timedelta(hours=1))
        request.assert_not_called()
        self.assertEqual(TradeData.objects.filter(symbol=symbol).count(), 1)

    def test_dispatch_rejects_perpetual_symbols(self):
        symbol = self.get_symbol(
            exchange=Exchange.UPBIT,
            api_symbol="KRW-BTC",
            symbol_type=SymbolType.PERPETUAL,
        )
        with self.assertRaises(ValueError):
            trades_api(symbol, START, START + timedelta(minutes=1), Mock())
        with self.assertRaises(ValueError):
            candles_api(symbol, START, START + timedelta(minutes=1))
