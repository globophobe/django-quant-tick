import random
from datetime import datetime
from typing import List, Tuple

import pandas as pd
from django.test import TestCase

from cryptofeed_werks.constants import SymbolType
from cryptofeed_werks.lib import (
    aggregate_trades,
    get_current_time,
    get_min_time,
    volume_filter_with_time_window,
)
from cryptofeed_werks.models import AggregatedTrade, Candle, GlobalSymbol, Symbol

from .base import RandomTradeTestCase


class BaseCandleTestCase(TestCase):
    def setUp(self):
        self.timestamp_from = get_min_time(get_current_time(), "1d")
        global_symbol = GlobalSymbol.objects.create(name="test")
        self.symbol = Symbol.objects.create(
            global_symbol=global_symbol,
            symbol_type=random.choice(SymbolType.values),
            name="test",
        )


class IterAllTestCase(BaseCandleTestCase):
    def setUp(self):
        super().setUp()
        self.one_minute = pd.Timedelta("1t")
        self.timestamp_to = self.timestamp_from + (self.one_minute * 5)

    def get_values(self, retry: bool = False) -> List[Tuple[datetime, datetime]]:
        """Get values."""
        return [
            value
            for value in Candle.iter_all(
                symbol=self.symbol,
                timestamp_from=self.timestamp_from,
                timestamp_to=self.timestamp_to,
                retry=retry,
                reverse=True,
            )
        ]

    def test_iter_all_with_head(self):
        """First candle is OK."""
        Candle.objects.create(
            symbol=self.symbol, timestamp=self.timestamp_from, ok=True
        )
        values = self.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from + self.one_minute)
        self.assertEqual(values[-1][1], self.timestamp_to)

    def test_iter_all_with_one_candle_ok(self):
        """Second candle is OK."""
        candle = Candle.objects.create(
            symbol=self.symbol, timestamp=self.timestamp_from + self.one_minute, ok=True
        )
        values = self.get_values()
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0][0], candle.timestamp + self.one_minute)
        self.assertEqual(values[0][1], self.timestamp_to)
        self.assertEqual(values[-1][0], self.timestamp_from)
        self.assertEqual(values[-1][1], candle.timestamp)

    def test_iter_all_with_two_candles_ok(self):
        """Second and fourth candles are OK."""
        candle_one = Candle.objects.create(
            symbol=self.symbol, timestamp=self.timestamp_from + self.one_minute, ok=True
        )
        candle_two = Candle.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from + (self.one_minute * 3),
            ok=True,
        )
        values = self.get_values()
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0][0], self.timestamp_to - self.one_minute)
        self.assertEqual(values[0][1], self.timestamp_to)
        self.assertEqual(values[1][0], candle_one.timestamp + self.one_minute)
        self.assertEqual(values[1][1], candle_two.timestamp)
        self.assertEqual(values[-1][0], self.timestamp_from)
        self.assertEqual(values[-1][1], candle_one.timestamp)

    def test_iter_all_with_tail(self):
        """Last candle is OK."""
        Candle.objects.create(
            symbol=self.symbol, timestamp=self.timestamp_to - self.one_minute, ok=True
        )
        values = self.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[0][1], self.timestamp_to - self.one_minute)

    def test_iter_all_with_retry_and_one_candle_not_ok(self):
        """One candle not OK."""
        Candle.objects.create(
            symbol=self.symbol, timestamp=self.timestamp_from, ok=False
        )
        values = self.get_values(retry=True)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[-1][1], self.timestamp_to)

    def test_iter_all_with_retry_and_one_candle_missing(self):
        """One candle missing."""
        Candle.objects.create(
            symbol=self.symbol, timestamp=self.timestamp_from, ok=None
        )
        values = self.get_values(retry=True)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from + self.one_minute)
        self.assertEqual(values[-1][1], self.timestamp_to)


class WriteCandleTestCase(RandomTradeTestCase, BaseCandleTestCase):
    def setUp(self):
        super().setUp()
        self.timestamp_to = self.timestamp_from + pd.Timedelta("1t")

    def test_write_candle(self):
        """Write candles from aggregate trades."""
        trades = [self.get_random_trade(timestamp=self.timestamp_from)]
        data_frame = pd.DataFrame(trades)
        aggregated = aggregate_trades(data_frame)
        filtered = volume_filter_with_time_window(
            aggregated, min_volume=None, window="1t"
        )
        Candle.write(
            self.symbol,
            self.timestamp_from,
            self.timestamp_to,
            filtered,
        )
        row = filtered.iloc[0]
        # Candles
        candles = Candle.objects.all()
        self.assertEqual(candles.count(), 1)
        candle = candles[0]
        candles = Candle.objects.all()
        self.assertEqual(candle.symbol, self.symbol)
        self.assertEqual(candle.uid, row.uid)
        self.assertEqual(candle.timestamp, row.timestamp)
        for attr in ("open", "high", "low", "close"):
            self.assertEqual(getattr(candle, attr), row.price)
        self.assertEqual(candle.buy_volume, row.totalBuyVolume)
        self.assertEqual(candle.volume, row.totalVolume)
        self.assertEqual(candle.buy_notional, row.totalBuyNotional)
        self.assertEqual(candle.notional, row.totalNotional)
        self.assertEqual(candle.buy_ticks, row.totalBuyTicks)
        self.assertEqual(candle.ticks, row.totalTicks)
        self.assertFalse(candle.ok)
        # Aggregated trades
        aggregated_trades = AggregatedTrade.objects.all()
        self.assertEqual(aggregated_trades.count(), 1)
        aggregated_trade = aggregated_trades[0]
        self.assertEqual(aggregated_trade.candle, candle)
        self.assertEqual(aggregated_trade.timestamp, row.timestamp)
        self.assertEqual(aggregated_trade.nanoseconds, 0)
        for attr in ("price", "high", "low"):
            self.assertEqual(getattr(aggregated_trade, attr), row.price)
        self.assertEqual(aggregated_trade.volume, row.volume)
        self.assertEqual(aggregated_trade.notional, row.notional)
        self.assertEqual(aggregated_trade.tick_rule, row.tickRule)
        self.assertEqual(aggregated_trade.ticks, row.ticks)
        self.assertEqual(aggregated_trade.total_buy_volume, row.totalBuyVolume)
        self.assertEqual(aggregated_trade.total_volume, row.totalVolume)
        self.assertEqual(aggregated_trade.total_buy_notional, row.totalBuyNotional)
        self.assertEqual(aggregated_trade.total_notional, row.totalNotional)
        self.assertEqual(aggregated_trade.total_buy_ticks, row.totalBuyTicks)
        self.assertEqual(aggregated_trade.total_ticks, row.totalTicks)
