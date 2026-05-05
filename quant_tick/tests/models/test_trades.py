import os
import shutil
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

    def test_write_trade_data(self):
        symbol = self.get_symbol()
        raw = self.get_raw(self.timestamp_from)
        TradeData.write(
            symbol, self.timestamp_from, self.timestamp_to, raw, pd.DataFrame([])
        )
        row = raw.iloc[0]
        trade_data = TradeData.objects.all()
        self.assertEqual(trade_data.count(), 1)
        t = trade_data[0]
        self.assertEqual(t.symbol, symbol)
        self.assertEqual(t.uid, row.uid)
        self.assertEqual(t.timestamp, row.timestamp)
        self.assertFalse(t.ok)

    def test_retry_raw_trade(self):
        symbol = self.get_symbol(save_raw=True)
        raw = self.get_raw(self.timestamp_from)
        for i in range(2):
            TradeData.write(
                symbol,
                self.timestamp_from,
                self.timestamp_to,
                raw,
                pd.DataFrame([]),
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
                aggregated,
                pd.DataFrame([]),
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
            self.get_raw(day_from),
            pd.DataFrame([]),
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
            self.get_raw(hour_from),
            pd.DataFrame([]),
        )

        rows = list(TradeData.objects.filter(symbol=symbol).order_by("timestamp", "frequency"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].timestamp, hour_from)
        self.assertEqual(rows[0].frequency, Frequency.HOUR)

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
            TradeData.write(symbol, ts_from, ts_to, df, pd.DataFrame([]))
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
            TradeData.write(symbol, ts_from, ts_to, df, pd.DataFrame([]))

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
