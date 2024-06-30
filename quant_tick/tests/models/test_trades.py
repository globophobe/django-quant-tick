import os
from pathlib import Path

import pandas as pd
from django.test import TestCase

from quant_tick.constants import FileData
from quant_tick.lib import get_min_time, get_next_time
from quant_tick.models import TradeData
from quant_tick.storage import convert_trade_data_to_hourly

from ..base import BaseWriteTradeDataTest


class WriteTradeDataTest(BaseWriteTradeDataTest, TestCase):
    def setUp(self):
        super().setUp()
        self.timestamp_to = self.timestamp_from + pd.Timedelta("1min")

    def test_write_trade_data(self):
        """Write trade data."""
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
        """Retry raw trade."""
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
        """Retry aggregated trade."""
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

    def test_convert_trade_data_to_hourly(self):
        """Convert trade data to hourly."""
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

        convert_trade_data_to_hourly(
            symbol, timestamp_from, get_next_time(timestamp_from, value="1h")
        )

        trades = TradeData.objects.all()
        self.assertEqual(trades.count(), 1)

        raw = pd.concat(data_frames).drop(columns=["uid"]).reset_index(drop=True)
        data = trades[0]
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
