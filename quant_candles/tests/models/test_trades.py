import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from django.test import TestCase

from quant_candles.lib import get_min_time
from quant_candles.models import TradeData
from quant_candles.storage import convert_trade_data_to_hourly

from ..base import BaseWriteTradeDataTest


class WriteTradeDataTest(BaseWriteTradeDataTest, TestCase):
    def setUp(self):
        super().setUp()
        self.timestamp_to = self.timestamp_from + pd.Timedelta("1t")

    def test_write_trade_data(self):
        """Write trade data."""
        symbol = self.get_symbol()
        filtered = self.get_filtered(self.timestamp_from)
        TradeData.write(symbol, self.timestamp_from, self.timestamp_to, filtered, {})
        row = filtered.iloc[0]
        trade_data = TradeData.objects.all()
        self.assertEqual(trade_data.count(), 1)
        t = trade_data[0]
        self.assertEqual(t.symbol, symbol)
        self.assertEqual(t.uid, row.uid)
        self.assertEqual(t.timestamp, row.timestamp)
        self.assertFalse(t.ok)

    def test_retry_raw_trade(self):
        """Retry raw trade."""
        symbol = self.get_symbol(should_aggregate_trades=False)
        filtered = self.get_filtered(self.timestamp_from)
        for i in range(2):
            TradeData.write(
                symbol, self.timestamp_from, self.timestamp_to, filtered, {}
            )
        trade_data = TradeData.objects.all()
        self.assertEqual(trade_data.count(), 1)
        t = trade_data[0]
        _, filename = os.path.split(t.file_data.name)
        self.assertEqual(filename.count("."), 1)

        storage = t.file_data.storage
        exchange = t.symbol.exchange
        symbol = t.symbol.symbol
        path = Path("trades") / exchange / symbol / "raw"
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
        symbol = self.get_symbol()
        filtered = self.get_filtered(self.timestamp_from)
        for i in range(2):
            TradeData.write(
                symbol, self.timestamp_from, self.timestamp_to, filtered, {}
            )
        trade_data = TradeData.objects.all()
        self.assertEqual(trade_data.count(), 1)
        t = trade_data[0]
        _, filename = os.path.split(t.file_data.name)
        self.assertEqual(filename.count("."), 1)

        storage = t.file_data.storage
        exchange = t.symbol.exchange
        symbol = t.symbol.symbol
        path = Path("trades") / exchange / symbol / "aggregated" / "0"
        p = str(path.resolve())

        directories, _ = storage.listdir(p)
        self.assertEqual(len(directories), 1)
        directory = directories[0]
        d = path / directory
        _, files = storage.listdir(d)
        self.assertEqual(len(files), 1)
        fname = files[0]
        self.assertEqual(filename, fname)

    @patch("quant_candles.storage.candles_api")
    @patch("quant_candles.storage.validate_data_frame")
    def test_convert_trade_data_to_hourly(
        self, mock_validate_data_frame, mock_candle_api
    ):
        """Convert trade data to hourly."""
        symbol = self.get_symbol()
        timestamp_from = get_min_time(self.timestamp_from, "1h")

        data_frames = []
        for minute in range(60):
            ts_from = timestamp_from + pd.Timedelta(f"{minute}t")
            ts_to = ts_from + pd.Timedelta("1t")
            df = self.get_filtered(ts_from)
            validated = {minute: True}
            TradeData.write(symbol, ts_from, ts_to, df, validated)
            data_frames.append(df)

        first = TradeData.objects.get(timestamp=timestamp_from)

        mock_validate_data_frame.return_value = {
            timestamp: True
            for timestamp in [
                self.timestamp_from + pd.Timedelta(f"{minute}t") for minute in range(60)
            ]
        }

        convert_trade_data_to_hourly(symbol)

        trades = TradeData.objects.all()
        self.assertEqual(trades.count(), 1)

        filtered = pd.concat(data_frames).drop(columns=["uid"])
        data = trades[0]
        self.assertEqual(data.uid, first.uid)
        self.assertTrue(data.get_data_frame().equals(filtered))
        self.assertTrue(data.ok)
