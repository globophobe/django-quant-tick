import os
import shutil
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from django.test import TestCase

from quant_tick.constants import FileData, Frequency
from quant_tick.lib import get_min_time, get_next_time
from quant_tick.models import TradeData
from quant_tick.storage import (
    clean_trade_data_overlaps,
    convert_trade_data,
    convert_trade_data_to_daily,
)

from ..base import BaseWriteTradeDataTest


class WriteTradeDataTest(BaseWriteTradeDataTest, TestCase):
    def setUp(self):
        super().setUp()
        self.timestamp_to = self.timestamp_from + pd.Timedelta("1min")

    def get_raw_validation_data(self, uid: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "uid": uid,
                    "timestamp": self.timestamp_from + pd.Timedelta("10s"),
                    "nanoseconds": 0,
                    "price": Decimal("100"),
                    "volume": Decimal("1000"),
                    "notional": Decimal("10"),
                    "tickRule": 1,
                    "ticks": 1,
                }
            ]
        )

    def get_aggregated_validation_data(self, uid: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "uid": uid,
                    "timestamp": self.timestamp_from + pd.Timedelta("10s"),
                    "nanoseconds": 0,
                    "price": Decimal("100"),
                    "volume": Decimal("1000"),
                    "notional": Decimal("10"),
                    "tickRule": 1,
                    "ticks": 1,
                    "high": Decimal("101"),
                    "low": Decimal("99"),
                    "totalBuyVolume": Decimal("1000"),
                    "totalVolume": Decimal("1000"),
                    "totalBuyNotional": Decimal("10"),
                    "totalNotional": Decimal("10"),
                    "totalBuyTicks": 1,
                    "totalTicks": 1,
                }
            ]
        )

    def get_exchange_candles(self, notional: str) -> pd.DataFrame:
        return pd.DataFrame(
            [{"timestamp": self.timestamp_from, "notional": Decimal(notional)}]
        ).set_index("timestamp")

    def get_validation_cases(self):
        return (
            (
                "raw",
                {"save_raw": True},
                {"raw_trades": self.get_raw_validation_data("raw-1")},
                FileData.RAW,
            ),
            (
                "aggregated",
                {"save_aggregated": True},
                {
                    "aggregated_trades": self.get_aggregated_validation_data(
                        "aggregated-1"
                    )
                },
                FileData.AGGREGATED,
            ),
            (
                "significant",
                {"significant_trade_filter": 1000},
                {
                    "filtered_trades": self.get_aggregated_validation_data(
                        "significant-1"
                    )
                },
                FileData.FILTERED,
            ),
        )

    def test_write_trade_data(self):
        symbol = self.get_symbol()
        raw = self.get_raw(self.timestamp_from)
        TradeData.write(
            symbol,
            self.timestamp_from,
            self.timestamp_to,
            pd.DataFrame([]),
            raw_trades=raw,
        )
        row = raw.iloc[0]
        trade_data = TradeData.objects.all()
        self.assertEqual(trade_data.count(), 1)
        t = trade_data[0]
        self.assertEqual(t.symbol, symbol)
        self.assertEqual(t.uid, row.uid)
        self.assertEqual(t.timestamp, row.timestamp)
        self.assertFalse(t.ok)

    def test_write_trade_data_validates_exchange_candle(self):
        for name, symbol_kwargs, trade_kwargs, stored_data in self.get_validation_cases():
            with self.subTest(name=name):
                symbol = self.get_symbol(api_symbol=name, **symbol_kwargs)

                rows = TradeData.write(
                    symbol,
                    self.timestamp_from,
                    self.timestamp_to,
                    self.get_exchange_candles("10"),
                    **trade_kwargs,
                )

                self.assertEqual(len(rows), 1)
                trade_data = TradeData.objects.get(symbol=symbol)
                self.assertTrue(trade_data.ok)
                self.assertEqual(trade_data.uid, f"{name}-1")
                for file_data in FileData:
                    self.assertEqual(
                        trade_data.has_data_frame(file_data),
                        file_data == stored_data,
                    )
                self.assertEqual(
                    trade_data.json_data["candle"]["notional"],
                    Decimal("10"),
                )

    def test_write_trade_data_marks_mismatched_candle_not_ok(self):
        for name, symbol_kwargs, trade_kwargs, _stored_data in self.get_validation_cases():
            with self.subTest(name=name):
                symbol = self.get_symbol(api_symbol=name, **symbol_kwargs)

                rows = TradeData.write(
                    symbol,
                    self.timestamp_from,
                    self.timestamp_to,
                    self.get_exchange_candles("11"),
                    **trade_kwargs,
                )

                self.assertEqual(len(rows), 1)
                self.assertFalse(TradeData.objects.get(symbol=symbol).ok)

    def test_retry_raw_trade(self):
        symbol = self.get_symbol(save_raw=True)
        raw = self.get_raw(self.timestamp_from)
        for i in range(2):
            TradeData.write(
                symbol,
                self.timestamp_from,
                self.timestamp_to,
                pd.DataFrame([]),
                raw_trades=raw,
            )
        trade_data = TradeData.objects.all()
        self.assertEqual(trade_data.count(), 1)
        t = trade_data[0]
        filename = Path(t.raw_data.name).name
        self.assertEqual(filename.count("."), 1)

        storage = t.raw_data.storage
        path = Path("test-trades") / Path("/".join(t.symbol.upload_path)) / "raw"
        p = str(path.resolve())

        directories, _ = storage.listdir(p)
        self.assertEqual(len(directories), 1)
        directory = directories[0]
        d = path / directory
        _, files = storage.listdir(d)
        self.assertEqual(len(files), 1)
        fname = files[0]
        self.assertEqual(filename, fname)

    def test_retry_aggregated_trade(self):
        symbol = self.get_symbol(save_aggregated=True)
        aggregated = self.get_aggregated(self.timestamp_from)
        for i in range(2):
            TradeData.write(
                symbol,
                self.timestamp_from,
                self.timestamp_to,
                pd.DataFrame([]),
                aggregated_trades=aggregated,
            )
        trade_data = TradeData.objects.all()
        self.assertEqual(trade_data.count(), 1)
        t = trade_data[0]
        _, filename = os.path.split(t.aggregated_data.name)
        self.assertEqual(filename.count("."), 1)

        storage = t.aggregated_data.storage
        path = Path("test-trades") / Path("/".join(t.symbol.upload_path)) / "aggregated"
        p = str(path.resolve())

        directories, _ = storage.listdir(p)
        self.assertEqual(len(directories), 1)
        directory = directories[0]
        d = path / directory
        _, files = storage.listdir(d)
        self.assertEqual(len(files), 1)
        fname = files[0]
        self.assertEqual(filename, fname)

    def test_write_day_deletes_overlapping_hourly_and_minute_rows(self):
        symbol = self.get_symbol()
        day_from = get_min_time(self.timestamp_from, "1d")
        TradeData.objects.create(
            symbol=symbol,
            timestamp=day_from,
            frequency=Frequency.HOUR,
        )
        TradeData.objects.create(
            symbol=symbol,
            timestamp=day_from + pd.Timedelta("1min"),
            frequency=Frequency.MINUTE,
        )

        TradeData.write(
            symbol,
            day_from,
            day_from + pd.Timedelta("1d"),
            pd.DataFrame([]),
            raw_trades=self.get_raw(day_from),
        )

        rows = list(TradeData.objects.filter(symbol=symbol).order_by("timestamp", "frequency"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].timestamp, day_from)
        self.assertEqual(rows[0].frequency, Frequency.DAY)

    def test_write_hour_deletes_overlapping_daily_and_minute_rows(self):
        symbol = self.get_symbol()
        day_from = get_min_time(self.timestamp_from, "1d")
        hour_from = day_from + pd.Timedelta("3h")
        TradeData.objects.create(
            symbol=symbol,
            timestamp=day_from,
            frequency=Frequency.DAY,
        )
        TradeData.objects.create(
            symbol=symbol,
            timestamp=hour_from + pd.Timedelta("1min"),
            frequency=Frequency.MINUTE,
        )

        TradeData.write(
            symbol,
            hour_from,
            hour_from + pd.Timedelta("1h"),
            pd.DataFrame([]),
            raw_trades=self.get_raw(hour_from),
        )

        rows = list(TradeData.objects.filter(symbol=symbol).order_by("timestamp", "frequency"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].timestamp, hour_from)
        self.assertEqual(rows[0].frequency, Frequency.HOUR)

    def test_write_rejects_unaligned_timestamp_ranges(self):
        symbol = self.get_symbol()
        day_from = get_min_time(self.timestamp_from, "1d")
        cases = (
            (day_from + pd.Timedelta("30s"), pd.Timedelta("1d")),
            (day_from + pd.Timedelta("3h30s"), pd.Timedelta("1h")),
            (day_from + pd.Timedelta("3h30s"), pd.Timedelta("1min")),
        )

        for timestamp_from, delta in cases:
            with self.subTest(timestamp_from=timestamp_from, delta=delta):
                with self.assertRaises(ValueError):
                    TradeData.write(
                        symbol,
                        timestamp_from,
                        timestamp_from + delta,
                        pd.DataFrame([]),
                        raw_trades=self.get_raw(timestamp_from),
                    )

    def test_clean_trade_data_overlaps_deletes_hourly_and_minute_rows(self):
        symbol = self.get_symbol()
        day_from = get_min_time(self.timestamp_from, "1d")
        hour_from = day_from + pd.Timedelta("12h")
        minute_from = hour_from + pd.Timedelta("5min")

        TradeData.objects.create(
            symbol=symbol,
            timestamp=day_from,
            frequency=Frequency.DAY,
        )
        TradeData.objects.create(
            symbol=symbol,
            timestamp=hour_from,
            frequency=Frequency.HOUR,
        )
        TradeData.objects.create(
            symbol=symbol,
            timestamp=minute_from,
            frequency=Frequency.MINUTE,
        )

        deleted = clean_trade_data_overlaps(
            symbol,
            hour_from + pd.Timedelta("1min"),
            hour_from + pd.Timedelta("2min"),
        )

        rows = list(TradeData.objects.filter(symbol=symbol).order_by("timestamp", "frequency"))
        self.assertEqual(deleted, 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].timestamp, day_from)
        self.assertEqual(rows[0].frequency, Frequency.DAY)

    def test_clean_trade_data_overlaps_deletes_minute_rows_covered_by_hourly_row(self):
        symbol = self.get_symbol()
        day_from = get_min_time(self.timestamp_from, "1d")
        hour_from = day_from + pd.Timedelta("6h")
        minute_from = hour_from + pd.Timedelta("5min")

        TradeData.objects.create(
            symbol=symbol,
            timestamp=hour_from,
            frequency=Frequency.HOUR,
        )
        TradeData.objects.create(
            symbol=symbol,
            timestamp=minute_from,
            frequency=Frequency.MINUTE,
        )

        deleted = clean_trade_data_overlaps(
            symbol,
            minute_from,
            minute_from + pd.Timedelta("1min"),
        )

        rows = list(TradeData.objects.filter(symbol=symbol).order_by("timestamp", "frequency"))
        self.assertEqual(deleted, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].timestamp, hour_from)
        self.assertEqual(rows[0].frequency, Frequency.HOUR)

    def test_convert_trade_data_to_daily_compacts_complete_hour(self):
        symbol = self.get_symbol()
        timestamp_from = get_min_time(self.timestamp_from, "1h")

        data_frames = []
        for minute in range(60):
            ts_from = timestamp_from + pd.Timedelta(f"{minute}min")
            ts_to = ts_from + pd.Timedelta("1min")
            df = self.get_raw(ts_from)
            TradeData.write(symbol, ts_from, ts_to, pd.DataFrame([]), raw_trades=df)
            data_frames.append(df)

        trades = TradeData.objects.all()
        self.assertEqual(trades.count(), 60)
        first = trades[0]
        candles = pd.DataFrame([t.json_data["candle"] for t in trades])

        convert_trade_data_to_daily(
            symbol, timestamp_from, get_next_time(timestamp_from, value="1h")
        )

        trades = TradeData.objects.all()
        self.assertEqual(trades.count(), 1)

        raw = pd.concat(data_frames).drop(columns=["uid"]).reset_index(drop=True)
        data = trades[0]
        self.assertEqual(data.frequency, Frequency.HOUR)
        self.assertEqual(data.uid, first.uid)
        self.assertTrue(data.get_data_frame(FileData.RAW).equals(raw))
        candle = data.json_data["candle"]
        self.assertEqual(candle["timestamp"], candles.iloc[0].timestamp)
        self.assertEqual(candle["open"], candles.iloc[0].open)
        self.assertEqual(candle["high"], candles.high.max())
        self.assertEqual(candle["low"], candles.low.min())
        self.assertEqual(candle["close"], candles.iloc[-1].close)
        self.assertEqual(candle["volume"], candles.volume.sum())
        self.assertEqual(candle["buyVolume"], candles.buyVolume.sum())
        self.assertEqual(candle["notional"], candles.notional.sum())
        self.assertEqual(candle["buyNotional"], candles.buyNotional.sum())
        self.assertEqual(candle["ticks"], candles.ticks.sum())
        self.assertEqual(candle["buyTicks"], candles.buyTicks.sum())

    def test_convert_trade_data_to_daily_compacts_complete_day(self):
        symbol = self.get_symbol(save_raw=False)
        day_from = get_min_time(self.timestamp_from, "1d")

        for hour in range(24):
            TradeData.objects.create(
                symbol=symbol,
                timestamp=day_from + pd.Timedelta(f"{hour}h"),
                frequency=Frequency.HOUR,
                ok=True,
            )

        convert_trade_data_to_daily(symbol, day_from, day_from + pd.Timedelta("1d"))

        trades = list(TradeData.objects.filter(symbol=symbol))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].timestamp, day_from)
        self.assertEqual(trades[0].frequency, Frequency.DAY)

    def test_convert_trade_data_to_daily_does_not_compact_incomplete_day(self):
        symbol = self.get_symbol(save_raw=False)
        day_from = get_min_time(self.timestamp_from, "1d")

        for hour in range(4):
            TradeData.objects.create(
                symbol=symbol,
                timestamp=day_from
                + pd.Timedelta(f"{hour}h")
                + pd.Timedelta("30min"),
                frequency=Frequency.HOUR,
                ok=True,
            )
        for hour in range(4, 24):
            TradeData.objects.create(
                symbol=symbol,
                timestamp=day_from + pd.Timedelta(f"{hour}h"),
                frequency=Frequency.HOUR,
                ok=True,
            )

        convert_trade_data_to_daily(symbol, day_from, day_from + pd.Timedelta("1d"))

        trades = TradeData.objects.filter(symbol=symbol)
        self.assertFalse(trades.filter(frequency=Frequency.DAY).exists())
        self.assertEqual(trades.filter(frequency=Frequency.HOUR).count(), 24)

    def test_convert_trade_data_deletes_source_rows_before_write(self):
        symbol = self.get_symbol()
        timestamp_from = get_min_time(self.timestamp_from, "1h")

        for minute in range(60):
            ts_from = timestamp_from + pd.Timedelta(f"{minute}min")
            ts_to = ts_from + pd.Timedelta("1min")
            df = self.get_raw(ts_from)
            TradeData.write(symbol, ts_from, ts_to, pd.DataFrame([]), raw_trades=df)

        queryset = TradeData.objects.filter(
            symbol=symbol,
            timestamp__gte=timestamp_from,
            timestamp__lt=get_next_time(timestamp_from, value="1h"),
            frequency=Frequency.MINUTE,
        )

        try:
            with patch(
                "quant_tick.models.trades.TradeData.save",
                side_effect=RuntimeError("boom"),
            ):
                with self.assertRaises(RuntimeError):
                    convert_trade_data(
                        symbol,
                        queryset,
                        timestamp_from,
                        get_next_time(timestamp_from, value="1h"),
                    )

            self.assertFalse(
                TradeData.objects.filter(symbol=symbol, frequency=Frequency.MINUTE).exists()
            )
            self.assertFalse(
                TradeData.objects.filter(symbol=symbol, frequency=Frequency.HOUR).exists()
            )
        finally:
            shutil.rmtree(Path("test-trades"), ignore_errors=True)
