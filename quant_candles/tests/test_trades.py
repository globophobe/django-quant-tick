import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import pandas as pd
import time_machine
from django.test import TestCase
from pandas import DataFrame

from quant_candles.constants import Exchange, Frequency
from quant_candles.lib import (
    aggregate_trades,
    get_current_time,
    get_min_time,
    volume_filter_with_time_window,
)
from quant_candles.models import GlobalSymbol, Symbol, TradeData
from quant_candles.storage import convert_trade_data_to_hourly

from .base import RandomTradeTest


class BaseTradeDataTest(TestCase):
    def setUp(self):
        self.timestamp_from = get_min_time(get_current_time(), "1d")
        self.global_symbol = GlobalSymbol.objects.create(name="global-symbol")

    def get_symbol(
        self, api_symbol: str = "test", should_aggregate_trades: bool = True
    ) -> Symbol:
        """Get symbol."""
        return Symbol.objects.create(
            global_symbol=self.global_symbol,
            exchange=Exchange.FTX,
            api_symbol=api_symbol,
            should_aggregate_trades=should_aggregate_trades,
        )


@time_machine.travel(datetime(2009, 1, 3))
class IterAllTest(BaseTradeDataTest):
    def setUp(self):
        super().setUp()
        self.one_minute = pd.Timedelta("1t")
        self.timestamp_to = self.timestamp_from + (self.one_minute * 5)
        self.symbol = self.get_symbol()

    def get_values(self, retry: bool = False) -> List[Tuple[datetime, datetime]]:
        """Get values."""
        return [
            value
            for value in TradeData.iter_all(
                symbol=self.symbol,
                timestamp_from=self.timestamp_from,
                timestamp_to=self.timestamp_to,
                retry=retry,
                reverse=True,
            )
        ]

    @patch(
        "quant_candles.models.trades.TradeData.get_max_timestamp_to",
        return_value=datetime(2009, 1, 3).replace(tzinfo=timezone.utc),
    )
    def test_iter_all_with_no_results(self, mock_get_max_timestamp_to):
        """No results."""
        values = self.get_values()
        self.assertEqual(len(values), 0)

    @patch(
        "quant_candles.models.trades.TradeData.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=timezone.utc),
    )
    def test_iter_all_with_head(self, mock_get_max_timestamp_to):
        """First is OK."""
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from,
            frequency=Frequency.MINUTE,
            ok=True,
        )
        values = self.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from + self.one_minute)
        self.assertEqual(values[-1][1], self.timestamp_to)

    @patch(
        "quant_candles.models.trades.TradeData.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=timezone.utc),
    )
    def test_iter_all_with_one_ok(self, mock_get_max_timestamp_to):
        """Second is OK."""
        obj = TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from + self.one_minute,
            frequency=Frequency.MINUTE,
            ok=True,
        )
        values = self.get_values()
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0][0], obj.timestamp + self.one_minute)
        self.assertEqual(values[0][1], self.timestamp_to)
        self.assertEqual(values[-1][0], self.timestamp_from)
        self.assertEqual(values[-1][1], obj.timestamp)

    @patch(
        "quant_candles.models.trades.TradeData.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=timezone.utc),
    )
    def test_iter_all_with_two_ok(self, mock_get_max_timestamp_to):
        """Second and fourth are OK."""
        obj_one = TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from + self.one_minute,
            frequency=Frequency.MINUTE,
            ok=True,
        )
        obj_two = TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from + (self.one_minute * 3),
            frequency=Frequency.MINUTE,
            ok=True,
        )
        values = self.get_values()
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0][0], self.timestamp_to - self.one_minute)
        self.assertEqual(values[0][1], self.timestamp_to)
        self.assertEqual(values[1][0], obj_one.timestamp + self.one_minute)
        self.assertEqual(values[1][1], obj_two.timestamp)
        self.assertEqual(values[-1][0], self.timestamp_from)
        self.assertEqual(values[-1][1], obj_one.timestamp)

    @patch(
        "quant_candles.models.trades.TradeData.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=timezone.utc),
    )
    def test_iter_all_with_tail(self, mock_get_max_timestamp_to):
        """Last is OK."""
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_to - self.one_minute,
            frequency=Frequency.MINUTE,
            ok=True,
        )
        values = self.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[0][1], self.timestamp_to - self.one_minute)

    @patch(
        "quant_candles.models.trades.TradeData.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=timezone.utc),
    )
    def test_iter_all_with_retry_and_one_not_ok(self, mock_get_max_timestamp_to):
        """One is not OK."""
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from,
            frequency=Frequency.MINUTE,
            ok=False,
        )
        values = self.get_values(retry=True)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[-1][1], self.timestamp_to)

    @patch(
        "quant_candles.models.trades.TradeData.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=timezone.utc),
    )
    def test_iter_all_with_retry_and_one_missing(self, mock_get_max_timestamp_to):
        """One is missing."""
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from,
            frequency=Frequency.MINUTE,
            ok=None,
        )
        values = self.get_values(retry=True)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from + self.one_minute)
        self.assertEqual(values[-1][1], self.timestamp_to)


class WriteTradeDataTest(RandomTradeTest, BaseTradeDataTest):
    def setUp(self):
        super().setUp()
        self.timestamp_to = self.timestamp_from + pd.Timedelta("1t")

    def get_filtered(self, timestamp: datetime) -> DataFrame:
        """Get filtered."""
        trades = [self.get_random_trade(timestamp=timestamp)]
        data_frame = pd.DataFrame(trades)
        aggregated = aggregate_trades(data_frame)
        return volume_filter_with_time_window(aggregated, min_volume=None, window="1t")

    def tearDown(self):
        trade_data = TradeData.objects.select_related("symbol")
        # Files
        for obj in trade_data:
            obj.delete()
        # Directories
        for obj in trade_data:
            storage = obj.file_data.storage
            trades = Path("trades")
            exchange = obj.symbol.exchange
            symbol = obj.symbol.symbol
            head = trades / exchange / symbol
            if obj.symbol.should_aggregate_trades:
                tail = Path("aggregated") / str(obj.symbol.significant_trade_filter)
            else:
                tail = "raw"
            path = head / tail
            directories, _ = storage.listdir(str(path.resolve()))
            for directory in directories:
                storage.delete(path / directory)
            storage.delete(path)
            storage.delete(trades / exchange / symbol / "aggregated")
            storage.delete(trades / exchange / symbol)
            storage.delete(trades / exchange)
            storage.delete(trades)

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
