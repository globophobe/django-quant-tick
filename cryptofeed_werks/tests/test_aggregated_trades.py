from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from django.test import TestCase

from cryptofeed_werks.lib import (
    aggregate_trades,
    get_current_time,
    get_min_time,
    volume_filter_with_time_window,
)
from cryptofeed_werks.models import AggregatedTradeData, GlobalSymbol, Symbol

from .base import RandomTradeTestCase


class BaseAggregatedTradeTestCase(TestCase):
    def setUp(self):
        self.timestamp_from = get_min_time(get_current_time(), "1d")
        global_symbol = GlobalSymbol.objects.create(name="global-symbol")
        self.symbol = Symbol.objects.create(
            global_symbol=global_symbol, exchange="exchange", api_symbol="symbol"
        )


class IterAllTestCase(BaseAggregatedTradeTestCase):
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
            symbol=self.symbol, timestamp=self.timestamp_from, ok=True
        )
        values = self.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from + self.one_minute)
        self.assertEqual(values[-1][1], self.timestamp_to)

    def test_iter_all_with_one_aggregated_trade_ok(self):
        """Second is OK."""
        obj = AggregatedTradeData.objects.create(
            symbol=self.symbol, timestamp=self.timestamp_from + self.one_minute, ok=True
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
            symbol=self.symbol, timestamp=self.timestamp_from + self.one_minute, ok=True
        )
        obj_two = AggregatedTradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from + (self.one_minute * 3),
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
            symbol=self.symbol, timestamp=self.timestamp_to - self.one_minute, ok=True
        )
        values = self.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[0][1], self.timestamp_to - self.one_minute)

    def test_iter_all_with_retry_and_one_aggregated_trade_not_ok(self):
        """One is not OK."""
        AggregatedTradeData.objects.create(
            symbol=self.symbol, timestamp=self.timestamp_from, ok=False
        )
        values = self.get_values(retry=True)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[-1][1], self.timestamp_to)

    def test_iter_all_with_retry_and_one_aggregated_trade_missing(self):
        """One is missing."""
        AggregatedTradeData.objects.create(
            symbol=self.symbol, timestamp=self.timestamp_from, ok=None
        )
        values = self.get_values(retry=True)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from + self.one_minute)
        self.assertEqual(values[-1][1], self.timestamp_to)


class WriteAggregregatedTradeDataTestCase(
    RandomTradeTestCase, BaseAggregatedTradeTestCase
):
    def setUp(self):
        super().setUp()
        self.timestamp_to = self.timestamp_from + pd.Timedelta("1t")

    def tearDown(self):
        aggregate_trades = AggregatedTradeData.objects.select_related("symbol")
        # Files
        for obj in aggregate_trades:
            obj.delete()
        # Directories
        for obj in aggregate_trades:
            storage = obj.data.storage
            exchange = obj.symbol.exchange
            symbol = obj.symbol.symbol
            sym = Path(exchange) / symbol
            s = str(sym.resolve())
            directories, _ = storage.listdir(s)
            for directory in directories:
                d = sym / directory
                storage.delete(str(d.resolve()))
            storage.delete(s)
            storage.delete(exchange)

    def test_write_aggregated_trade(self):
        """Write aggregated trade."""
        trades = [self.get_random_trade(timestamp=self.timestamp_from)]
        data_frame = pd.DataFrame(trades)
        aggregated = aggregate_trades(data_frame)
        filtered = volume_filter_with_time_window(
            aggregated, min_volume=None, window="1t"
        )
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
