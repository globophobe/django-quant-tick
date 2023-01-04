from datetime import datetime, timezone
from typing import List, Tuple
from unittest.mock import patch

import pandas as pd
import time_machine
from django.test import TestCase, override_settings
from pandas import DataFrame

from quant_candles.constants import Frequency
from quant_candles.controllers import CandleCacheIterator, aggregate_candles
from quant_candles.lib import get_current_time, get_next_monday
from quant_candles.models import (
    Candle,
    CandleCache,
    CandleData,
    CandleReadOnlyData,
    TimeBasedCandle,
    TradeData,
)

from ..base import BaseSymbolTest, BaseWriteTradeDataTest


class BaseCandleCacheIteratorTest(BaseSymbolTest):
    def setUp(self):
        super().setUp()
        self.candle = self.get_candle()
        self.symbol = self.get_symbol()
        self.candle.symbols.add(self.symbol)

    def get_candle(self) -> Candle:
        """Get candle."""
        return Candle.objects.create()

    def get_values(self, step: str = "1d") -> List[Tuple[datetime, datetime]]:
        """Get values."""
        return [
            value
            for value in CandleCacheIterator(self.candle).iter_all(
                self.timestamp_from, self.timestamp_to, step
            )
        ]


class BaseCandleCacheWeeklyIteratorTest(BaseCandleCacheIteratorTest):
    def setUp(self):
        super().setUp()
        self.timestamp_from = get_next_monday(get_current_time())
        self.timestamp_to = get_next_monday(self.timestamp_from)
        self.one_hour = pd.Timedelta("1h")


@time_machine.travel(datetime(2009, 1, 3))
class CandleCacheIteratorTest(BaseCandleCacheIteratorTest, TestCase):
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


@time_machine.travel(datetime(2009, 1, 3))
class CandleCacheWeeklyIteratorTest(BaseCandleCacheWeeklyIteratorTest, TestCase):
    def test_iter_all_with_seven_day_step(self):
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

    def test_iter_all_with_seven_day_step_and_missing_trade_data(self):
        """From Monday, to Monday after next. Trade data is missing, so no values."""
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from,
            frequency=Frequency.WEEK.value - self.one_hour.total_seconds() / 60,
        )
        values = self.get_values(step="7d")
        self.assertEqual(len(values), 0)


@time_machine.travel(datetime(2009, 1, 3))
@patch(
    "quant_candles.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 4).replace(tzinfo=timezone.utc),
)
class TimeBasedMinuteFrequencyCandleTest(
    BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):

    databases = {"default", "read_only"}

    def get_candle(self) -> Candle:
        """Get candle."""
        return TimeBasedCandle.objects.create(json_data={"window": "1t"})

    def write_trade_data(self, data_frame: DataFrame) -> None:
        """Write trade data."""
        trade_data = TradeData(
            symbol=self.symbol,
            timestamp=data_frame.iloc[0].timestamp,
            frequency=Frequency.MINUTE,
        )
        TradeData.write_data_frame(trade_data, data_frame, pd.DataFrame([]))

    def test_one_candle_from_trade_in_the_first_minute(self, mock_get_max_timestamp_to):
        """One candle from a trade in the first minute."""
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(filtered)
        aggregate_candles(
            self.candle, self.timestamp_from, self.timestamp_from + pd.Timedelta("1t")
        )
        candle_data = CandleReadOnlyData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    @override_settings(IS_LOCAL=False)
    def test_one_candle_from_trade_in_the_first_minute_not_read_only(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade in the first minute, which is not read only."""
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(filtered)
        aggregate_candles(
            self.candle, self.timestamp_from, self.timestamp_from + pd.Timedelta("1t")
        )
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_two_candles_from_trades_in_the_first_and_second_minute(
        self, mock_get_max_timestamp_to
    ):
        """Two candles, one in first minute, and another in the second minute."""
        filtered_1 = self.get_filtered(self.timestamp_from)
        filtered_2 = self.get_filtered(self.timestamp_from + pd.Timedelta("1t"))
        self.write_trade_data(filtered_1)
        self.write_trade_data(filtered_2)
        aggregate_candles(
            self.candle, self.timestamp_from, self.timestamp_from + pd.Timedelta("2t")
        )
        candle_data = CandleReadOnlyData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, filtered_1.iloc[0].timestamp)
        self.assertEqual(candle_data[1].timestamp, filtered_2.iloc[0].timestamp)

    def test_two_candles_from_trades_in_first_and_third_minute(
        self, mock_get_max_timestamp_to
    ):
        """
        Two candles, one in first minute, and another in the third minute.
        No trades, no candle, for the second minute.
        """
        filtered_1 = self.get_filtered(self.timestamp_from)
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from + pd.Timedelta("1t"),
            frequency=Frequency.MINUTE,
        )
        filtered_2 = self.get_filtered(self.timestamp_from + pd.Timedelta("2t"))
        self.write_trade_data(filtered_1)
        self.write_trade_data(filtered_2)
        aggregate_candles(
            self.candle, self.timestamp_from, self.timestamp_from + pd.Timedelta("3t")
        )
        candle_data = CandleReadOnlyData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, filtered_1.iloc[0].timestamp)
        self.assertEqual(candle_data[1].timestamp, filtered_2.iloc[0].timestamp)


@time_machine.travel(datetime(2009, 1, 3))
@patch(
    "quant_candles.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 4).replace(tzinfo=timezone.utc),
)
class TimeBasedHourFrequencyCandleTest(
    BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):

    databases = {"default", "read_only"}

    def get_candle(self) -> Candle:
        """Get candle."""
        return TimeBasedCandle.objects.create(json_data={"window": "1h"})

    def write_trade_data(self, data_frame: DataFrame) -> None:
        """Write trade data."""
        trade_data = TradeData(
            symbol=self.symbol,
            timestamp=data_frame.iloc[0].timestamp,
            frequency=Frequency.HOUR,
        )
        TradeData.write_data_frame(trade_data, data_frame, pd.DataFrame([]))

    def test_one_candle_from_trade_in_the_first_hour(self, mock_get_max_timestamp_to):
        """One candle from a trade in the first hour."""
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(filtered)
        aggregate_candles(
            self.candle, self.timestamp_from, self.timestamp_from + pd.Timedelta("1h")
        )
        candle_data = CandleReadOnlyData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    @override_settings(IS_LOCAL=False)
    def test_one_candle_from_trade_in_the_first_hour_not_read_only(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade in the first hour, which is not read only."""
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(filtered)
        aggregate_candles(
            self.candle, self.timestamp_from, self.timestamp_from + pd.Timedelta("1h")
        )
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_two_candles_from_trades_in_the_first_and_second_hour(
        self, mock_get_max_timestamp_to
    ):
        """Two candles, one in first hour, and another in the second hour."""
        filtered_1 = self.get_filtered(self.timestamp_from)
        filtered_2 = self.get_filtered(self.timestamp_from + pd.Timedelta("1h"))
        self.write_trade_data(filtered_1)
        self.write_trade_data(filtered_2)
        aggregate_candles(
            self.candle, self.timestamp_from, self.timestamp_from + pd.Timedelta("2h")
        )
        candle_data = CandleReadOnlyData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, filtered_1.iloc[0].timestamp)
        self.assertEqual(candle_data[1].timestamp, filtered_2.iloc[0].timestamp)

    def test_two_candles_from_trades_in_first_and_third_hour(
        self, mock_get_max_timestamp_to
    ):
        """
        Two candles, one in first hour, and another in the third hour.
        No trades, no candle, for the second hour.
        """
        filtered_1 = self.get_filtered(self.timestamp_from)
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from + pd.Timedelta("1h"),
            frequency=Frequency.HOUR,
        )
        filtered_2 = self.get_filtered(self.timestamp_from + pd.Timedelta("2h"))
        self.write_trade_data(filtered_1)
        self.write_trade_data(filtered_2)
        aggregate_candles(
            self.candle, self.timestamp_from, self.timestamp_from + pd.Timedelta("3h")
        )
        candle_data = CandleReadOnlyData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, filtered_1.iloc[0].timestamp)
        self.assertEqual(candle_data[1].timestamp, filtered_2.iloc[0].timestamp)
