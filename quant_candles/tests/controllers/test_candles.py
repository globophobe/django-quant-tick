from datetime import datetime, timezone
from typing import List, Tuple
from unittest.mock import patch

import pandas as pd
import time_machine

from quant_candles.constants import Frequency
from quant_candles.controllers import CandleCacheIterator
from quant_candles.lib import get_current_time, get_next_monday
from quant_candles.models import Candle, CandleCache, TradeData

from ..base import BaseSymbolTest


@time_machine.travel(datetime(2009, 1, 3))
class BaseCandleCacheIteratorTest(BaseSymbolTest):
    def setUp(self):
        super().setUp()
        self.candle = Candle.objects.create()
        self.symbol = self.get_symbol()
        self.candle.symbols.add(self.symbol)

    def get_values(self, step: str = "1d") -> List[Tuple[datetime, datetime]]:
        """Get values."""
        return [
            value
            for value in CandleCacheIterator(self.candle).iter_all(
                timestamp_from=self.timestamp_from,
                timestamp_to=self.timestamp_to,
                step=step,
            )
        ]


class CandleCacheIteratorTest(BaseCandleCacheIteratorTest):
    def setUp(self):
        super().setUp()
        self.one_minute = pd.Timedelta("1t")
        self.timestamp_to = self.timestamp_from + (self.one_minute * 5)

    def create_trade_data(self) -> None:
        """Create trade data."""
        for i in range(5):
            TradeData.objects.create(
                symbol=self.symbol,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}t"),
                frequency=Frequency.MINUTE,
            )

    @patch(
        "quant_candles.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 3).replace(tzinfo=timezone.utc),
    )
    def test_iter_all_with_no_results(self, mock_get_max_timestamp_to):
        """No results."""
        values = self.get_values()
        self.assertEqual(len(values), 0)

    @patch(
        "quant_candles.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=timezone.utc),
    )
    def test_iter_all_without_trade_data(self, mock_get_max_timestamp_to):
        """No results without trade data."""
        values = self.get_values()
        self.assertEqual(len(values), 0)

    @patch(
        "quant_candles.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=timezone.utc),
    )
    def test_iter_all_with_head(self, mock_get_max_timestamp_to):
        """First is OK."""
        CandleCache.objects.create(
            candle=self.candle,
            timestamp=self.timestamp_from,
            frequency=Frequency.MINUTE,
        )
        self.create_trade_data()
        values = self.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from + self.one_minute)
        self.assertEqual(values[-1][1], self.timestamp_to)

    @patch(
        "quant_candles.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=timezone.utc),
    )
    def test_iter_all_with_one_ok(self, mock_get_max_timestamp_to):
        """Second is OK."""
        obj = CandleCache.objects.create(
            candle=self.candle,
            timestamp=self.timestamp_from + self.one_minute,
            frequency=Frequency.MINUTE,
        )
        self.create_trade_data()
        values = self.get_values()
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[0][1], obj.timestamp)
        self.assertEqual(values[-1][0], obj.timestamp + self.one_minute)
        self.assertEqual(values[-1][1], self.timestamp_to)

    @patch(
        "quant_candles.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=timezone.utc),
    )
    def test_iter_all_with_two_ok(self, mock_get_max_timestamp_to):
        """Second and fourth are OK."""
        obj_one = CandleCache.objects.create(
            candle=self.candle,
            timestamp=self.timestamp_from + self.one_minute,
            frequency=Frequency.MINUTE,
        )
        obj_two = CandleCache.objects.create(
            candle=self.candle,
            timestamp=self.timestamp_from + (self.one_minute * 3),
            frequency=Frequency.MINUTE,
        )
        self.create_trade_data()
        values = self.get_values()
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[0][1], obj_one.timestamp)
        self.assertEqual(values[1][0], obj_one.timestamp + self.one_minute)
        self.assertEqual(values[1][1], obj_two.timestamp)
        self.assertEqual(values[-1][0], self.timestamp_to - self.one_minute)
        self.assertEqual(values[-1][1], self.timestamp_to)

    @patch(
        "quant_candles.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
        return_value=datetime(2009, 1, 4).replace(tzinfo=timezone.utc),
    )
    def test_iter_all_with_tail(self, mock_get_max_timestamp_to):
        """Last is OK."""
        CandleCache.objects.create(
            candle=self.candle,
            timestamp=self.timestamp_to - self.one_minute,
            frequency=Frequency.MINUTE,
        )
        self.create_trade_data()
        values = self.get_values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[0][1], self.timestamp_to - self.one_minute)


@patch(
    "quant_candles.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 31).replace(tzinfo=timezone.utc),
)
class CandleCacheWeeklyIteratorTest(BaseCandleCacheIteratorTest):
    def setUp(self):
        super().setUp()
        self.timestamp_from = get_next_monday(get_current_time())
        self.timestamp_to = get_next_monday(self.timestamp_from)
        self.one_hour = pd.Timedelta("1h")

    def test_iter_all_with_seven_day_step(self, mock_get_max_timestamp_to):
        """From Monday, to Monday after next."""
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from,
            frequency=Frequency.WEEK,
        )
        values = self.get_values(step="7d")
        self.assertEqual(len(values), 168)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[0][1], self.timestamp_from + self.one_hour)
        self.assertEqual(values[-1][0], self.timestamp_to - self.one_hour)
        self.assertEqual(values[-1][1], self.timestamp_to)

    def test_iter_all_with_seven_day_step_and_missing_trade_data(
        self, mock_get_max_timestamp_to
    ):
        """From Monday, to Monday after next. Trade data is missing, so no values."""
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from,
            frequency=Frequency.WEEK.value - self.one_hour.total_seconds() / 60,
        )
        values = self.get_values(step="7d")
        self.assertEqual(len(values), 0)
