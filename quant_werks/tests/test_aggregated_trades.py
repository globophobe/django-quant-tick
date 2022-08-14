import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import pandas as pd
from django.test import TestCase
from pandas import DataFrame

from quant_werks.constants import Exchange, Frequency
from quant_werks.lib import (
    aggregate_trades,
    get_current_time,
    get_min_time,
    volume_filter_with_time_window,
)
from quant_werks.models import AggregatedTradeData, GlobalSymbol, Symbol
from quant_werks.storage import convert_aggregated_to_hourly

from .base import RandomTradeTest


class BaseAggregatedTradeTest(TestCase):
    def setUp(self):
        self.timestamp_from = get_min_time(get_current_time(), "1d")
        global_symbol = GlobalSymbol.objects.create(name="global-symbol")
        self.symbol = Symbol.objects.create(
            global_symbol=global_symbol, exchange=Exchange.FTX, api_symbol="test"
        )


class IterAllTest(BaseAggregatedTradeTest):
    def setUp(self):
        super().setUp()
        self.one_minute = pd.Timedelta("1t")
        self.timestamp_to = self.timestamp_from + (self.one_minute * 5)

    def get_values(self, retry: bool = False) -> List[Tuple[datetime, datetime]]:
        """Get values."""
        return [
            value
            for value in AggregatedTradeData.iter_all(
                symbol=self.symbol,
                timestamp_from=self.timestamp_from,
                timestamp_to=self.timestamp_to,
                retry=retry,
                reverse=True,
            )
        ]

    def test_iter_all_with_head(self):
        """First is OK."""
        AggregatedTradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from,
            frequency=Frequency.MINUTE,
            ok=True,
        )
        values = self.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from + self.one_minute)
        self.assertEqual(values[-1][1], self.timestamp_to)

    def test_iter_all_with_one_aggregated_trade_ok(self):
        """Second is OK."""
        obj = AggregatedTradeData.objects.create(
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

    def test_iter_all_with_two_aggregated_trades_ok(self):
        """Second and fourth are OK."""
        obj_one = AggregatedTradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from + self.one_minute,
            frequency=Frequency.MINUTE,
            ok=True,
        )
        obj_two = AggregatedTradeData.objects.create(
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

    def test_iter_all_with_tail(self):
        """Last is OK."""
        AggregatedTradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_to - self.one_minute,
            frequency=Frequency.MINUTE,
            ok=True,
        )
        values = self.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[0][1], self.timestamp_to - self.one_minute)

    def test_iter_all_with_retry_and_one_aggregated_trade_not_ok(self):
        """One is not OK."""
        AggregatedTradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from,
            frequency=Frequency.MINUTE,
            ok=False,
        )
        values = self.get_values(retry=True)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[-1][1], self.timestamp_to)

    def test_iter_all_with_retry_and_one_aggregated_trade_missing(self):
        """One is missing."""
        AggregatedTradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from,
            frequency=Frequency.MINUTE,
            ok=None,
        )
        values = self.get_values(retry=True)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from + self.one_minute)
        self.assertEqual(values[-1][1], self.timestamp_to)


class WriteAggregregatedTradeDataTest(RandomTradeTest, BaseAggregatedTradeTest):
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
        aggregate_trades = AggregatedTradeData.objects.select_related("symbol")
        # Files
        for obj in aggregate_trades:
            obj.delete()
        # Directories
        for obj in aggregate_trades:
            storage = obj.data.storage
            trades = Path("trades")
            exchange = obj.symbol.exchange
            path = trades / exchange / obj.symbol.upload_symbol
            directories, _ = storage.listdir(str(path.resolve()))
            for directory in directories:
                storage.delete(path / directory)
            storage.delete(path)
            storage.delete(trades / exchange)
            storage.delete(trades)

    def test_write_aggregated_trade(self):
        """Write aggregated trade."""
        filtered = self.get_filtered(self.timestamp_from)
        AggregatedTradeData.write(
            self.symbol,
            self.timestamp_from,
            self.timestamp_to,
            filtered,
        )
        row = filtered.iloc[0]
        aggregated_trades = AggregatedTradeData.objects.all()
        self.assertEqual(aggregated_trades.count(), 1)
        aggregated_trade = aggregated_trades[0]
        self.assertEqual(aggregated_trade.symbol, self.symbol)
        self.assertEqual(aggregated_trade.uid, row.uid)
        self.assertEqual(aggregated_trade.timestamp, row.timestamp)
        self.assertFalse(aggregated_trade.ok)

    def test_retry_aggregated_trade(self):
        """Retry aggregated trade."""
        filtered = self.get_filtered(self.timestamp_from)
        for i in range(2):
            AggregatedTradeData.write(
                self.symbol, self.timestamp_from, self.timestamp_to, filtered
            )
        aggregated_trades = AggregatedTradeData.objects.all()
        self.assertEqual(aggregated_trades.count(), 1)
        aggregated_trade = aggregated_trades[0]
        _, filename = os.path.split(aggregated_trade.data.name)
        self.assertEqual(filename.count("."), 1)

        storage = aggregated_trade.data.storage
        exchange = aggregated_trade.symbol.exchange
        path = Path("trades") / exchange / aggregated_trade.symbol.upload_symbol
        p = str(path.resolve())

        directories, _ = storage.listdir(p)
        self.assertEqual(len(directories), 1)
        directory = directories[0]
        d = path / directory
        _, files = storage.listdir(d)
        self.assertEqual(len(files), 1)
        fname = files[0]
        self.assertEqual(filename, fname)

    @patch("quant_werks.storage.candles_api")
    @patch("quant_werks.storage.validate_data_frame")
    def test_convert_minute_to_hourly(self, mock_validate_data_frame, mock_candle_api):
        """Convert minute to hourly."""
        timestamp_from = get_min_time(self.timestamp_from, "1h")

        data_frames = []
        for minute in range(60):
            ts_from = timestamp_from + pd.Timedelta(f"{minute}t")
            ts_to = ts_from + pd.Timedelta("1t")
            df = self.get_filtered(ts_from)
            AggregatedTradeData.write(self.symbol, ts_from, ts_to, df)
            data_frames.append(df)

        mock_validate_data_frame.return_value = {
            timestamp: True
            for timestamp in [
                self.timestamp_from + pd.Timedelta(f"{minute}t") for minute in range(60)
            ]
        }

        convert_aggregated_to_hourly(self.symbol)

        aggregated_trades = AggregatedTradeData.objects.all()
        self.assertEqual(aggregated_trades.count(), 1)

        filtered = pd.concat(data_frames).drop(columns=["uid"])
        aggregated_trade = aggregated_trades[0]
        self.assertTrue(aggregated_trade.get_data_frame().equals(filtered))
