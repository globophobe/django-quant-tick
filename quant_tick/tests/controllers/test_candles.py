from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Tuple
from unittest.mock import patch

import pandas as pd
import time_machine
from django.db.models.functions import TruncDate
from django.test import TestCase, override_settings
from pandas import DataFrame

from quant_tick.constants import Frequency, SampleType
from quant_tick.controllers import CandleCacheIterator, aggregate_candles
from quant_tick.lib import (
    aggregate_candle,
    get_current_time,
    get_min_time,
    get_next_cache,
)
from quant_tick.models import (
    AdaptiveCandle,
    Candle,
    CandleCache,
    CandleData,
    CandleReadOnlyData,
    ConstantCandle,
    Symbol,
    TimeBasedCandle,
    TradeData,
    TradeDataSummary,
)

from ..base import BaseSymbolTest, BaseWriteTradeDataTest


class BaseMinuteIteratorTest:
    def setUp(self):
        super().setUp()
        self.timestamp_from = get_min_time(get_current_time(), "1d")
        self.one_minute_from_now = self.timestamp_from + pd.Timedelta("1t")
        self.two_minutes_from_now = self.timestamp_from + pd.Timedelta("2t")
        self.three_minutes_from_now = self.timestamp_from + pd.Timedelta("3t")


class BaseHourIteratorTest:
    def setUp(self):
        super().setUp()
        self.timestamp_from = get_min_time(get_current_time(), "1d")
        self.one_hour_from_now = self.timestamp_from + pd.Timedelta("1h")
        self.two_hours_from_now = self.timestamp_from + pd.Timedelta("2h")
        self.three_hours_from_now = self.timestamp_from + pd.Timedelta("3h")


class BaseDayIteratorTest:
    def setUp(self):
        super().setUp()
        self.timestamp_from = get_min_time(get_current_time(), "1d")
        self.one_day_from_now = self.timestamp_from + pd.Timedelta("1d")
        self.two_days_from_now = self.timestamp_from + pd.Timedelta("2d")
        self.three_days_from_now = self.timestamp_from + pd.Timedelta("3d")


class BaseCandleCacheIteratorTest(BaseSymbolTest):
    def setUp(self):
        super().setUp()
        self.candle = self.get_candle()
        self.symbol = self.get_symbol()
        self.candle.symbols.add(self.symbol)

    def get_candle(self) -> Candle:
        """Get candle."""
        return Candle.objects.create()

    def get_values(self) -> List[Tuple[datetime, datetime]]:
        """Get values."""
        return [
            value
            for value in CandleCacheIterator(self.candle).iter_all(
                self.timestamp_from, self.timestamp_to
            )
        ]


@override_settings(IS_LOCAL=False)
@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 4, 0, 5).replace(tzinfo=timezone.utc),
)
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

    def test_iter_all_no_results(self, mock_get_max_timestamp_to):
        """No results."""
        self.create_trade_data()
        for i in range(5):
            CandleCache.objects.create(
                candle=self.candle,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}t"),
                frequency=Frequency.MINUTE,
            )
        values = self.get_values()
        self.assertEqual(len(values), 0)

    def test_iter_all_without_trade_data(self, mock_get_max_timestamp_to):
        """No results without trade data."""
        values = self.get_values()
        self.assertEqual(len(values), 0)

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


@override_settings(IS_LOCAL=False)
@time_machine.travel(datetime(2009, 1, 4), tick=False)
class CandleTest(BaseDayIteratorTest, TestCase):
    databases = {"default", "read_only"}

    def setUp(self):
        super().setUp()
        self.candle = Candle.objects.create()

    def create_candle_cache(self, timestamp: datetime) -> CandleCache:
        """Create candle cache."""
        return CandleCache.objects.create(
            candle=self.candle, timestamp=timestamp, frequency=Frequency.DAY
        )

    def test_initial_timestamp_from_without_candle_date_from(self):
        """Initial timestamp from equals initial value."""
        timestamp_from, timestamp_to, _ = self.candle.initialize(
            self.timestamp_from, self.three_days_from_now
        )
        self.assertEqual(timestamp_from, self.timestamp_from)
        self.assertEqual(timestamp_to, self.three_days_from_now)

    def test_initial_timestamp_from_with_candle_date_from(self):
        """Initial timestamp from not less than candle date from."""
        self.candle.date_from = self.one_day_from_now.date()
        timestamp_from, timestamp_to, _ = self.candle.initialize(
            self.timestamp_from, self.three_days_from_now
        )
        self.assertEqual(timestamp_from, self.one_day_from_now)

    def test_initial_timestamp_from_with_candle_cache(self):
        """Initial timestamp from not less than candle cache timestamp."""
        for i in range(2):
            self.create_candle_cache(self.timestamp_from + pd.Timedelta(f"{i}d"))
        timestamp_from, timestamp_to, _ = self.candle.initialize(
            self.timestamp_from, self.three_days_from_now
        )
        self.assertEqual(timestamp_from, self.two_days_from_now)
        self.assertEqual(timestamp_to, self.three_days_from_now)

    def test_initial_timestamp_from_with_candle_cache_and_retry(self):
        """Initial timestamp not less than candle date from, excluding cache if retry."""
        for i in range(2):
            self.create_candle_cache(self.timestamp_from + pd.Timedelta(f"{i}d"))
        timestamp_from, timestamp_to, _ = self.candle.initialize(
            self.timestamp_from, self.three_days_from_now, retry=True
        )
        self.assertEqual(timestamp_from, self.timestamp_from)
        self.assertEqual(timestamp_to, self.three_days_from_now)

    def test_initial_timestamp_from_with_both_candle_date_from_and_candle_cache(self):
        """
        Initial timestamp from not less than both candle date from and
        candle cache timestamp.
        """
        self.candle.date_from = self.one_day_from_now.date()
        for i in range(2):
            self.create_candle_cache(self.timestamp_from + pd.Timedelta(f"{i}d"))
        timestamp_from, timestamp_to, _ = self.candle.initialize(
            self.timestamp_from, self.three_days_from_now
        )
        self.assertEqual(timestamp_from, self.two_days_from_now)
        self.assertEqual(timestamp_to, self.three_days_from_now)

    def test_initial_timestamp_from_with_both_candle_date_from_candle_cache_and_retry(
        self,
    ):
        """Initial timestamp not less than candle date from, excluding cache if retry."""
        self.candle.date_from = self.one_day_from_now.date()
        for i in range(3):
            self.create_candle_cache(self.timestamp_from + pd.Timedelta(f"{i}d"))
        timestamp_from, timestamp_to, _ = self.candle.initialize(
            self.timestamp_from, self.three_days_from_now, retry=True
        )
        self.assertEqual(timestamp_from, self.one_day_from_now)
        self.assertEqual(timestamp_to, self.three_days_from_now)


@override_settings(IS_LOCAL=False)
@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 4, 0, 3).replace(tzinfo=timezone.utc),
)
class TimeBasedMinuteFrequencyCandleTest(
    BaseMinuteIteratorTest,
    BaseWriteTradeDataTest,
    BaseCandleCacheIteratorTest,
    TestCase,
):
    databases = {"default", "read_only"}

    def get_candle(self) -> Candle:
        """Get candle."""
        return TimeBasedCandle.objects.create(json_data={"window": "1t"})

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        """Write trade data."""
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, validated={}
        )

    def test_one_candle_from_trade_in_the_first_minute(self, mock_get_max_timestamp_to):
        """One candle from a trade in the first minute."""
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, filtered)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_one_candle_from_trade_in_the_first_minute_with_retry(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade in the first minute with retry."""
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, filtered)
        for i in range(2):
            aggregate_candles(
                self.candle,
                self.timestamp_from,
                self.one_minute_from_now,
                retry=bool(i),
            )
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    @override_settings(IS_LOCAL=True)
    def test_one_candle_from_trade_in_the_first_minute_read_only(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade in the first minute, read only."""
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, filtered)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleReadOnlyData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_two_candles_from_trades_in_the_first_and_second_minute(
        self, mock_get_max_timestamp_to
    ):
        """Two candles, one in first minute, and another in the second minute."""
        filtered_1 = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, filtered_1)
        filtered_2 = self.get_filtered(self.one_minute_from_now)
        self.write_trade_data(
            self.one_minute_from_now, self.two_minutes_from_now, filtered_2
        )
        aggregate_candles(self.candle, self.timestamp_from, self.two_minutes_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        self.assertEqual(candle_data[1].timestamp, self.one_minute_from_now)

    def test_two_candles_from_trades_in_first_and_third_minute(
        self, mock_get_max_timestamp_to
    ):
        """
        Two candles, one in first minute, and another in the third minute.
        No trades, no candle, for the second minute.
        """
        filtered_1 = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, filtered_1)
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from + pd.Timedelta("1t"),
            frequency=Frequency.MINUTE,
        )
        filtered_2 = self.get_filtered(self.two_minutes_from_now)
        self.write_trade_data(
            self.two_minutes_from_now, self.three_minutes_from_now, filtered_2
        )
        aggregate_candles(self.candle, self.timestamp_from, self.three_minutes_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        self.assertEqual(candle_data[1].timestamp, self.two_minutes_from_now)


@override_settings(IS_LOCAL=False)
@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 4, 0, 3).replace(tzinfo=timezone.utc),
)
class TimeBasedTwoMinuteFrequencyCandleTest(
    BaseMinuteIteratorTest,
    BaseWriteTradeDataTest,
    BaseCandleCacheIteratorTest,
    TestCase,
):
    databases = {"default", "read_only"}

    def get_candle(self) -> Candle:
        """Get candle."""
        return TimeBasedCandle.objects.create(json_data={"window": "2t"})

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        """Write trade data."""
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, validated={}
        )

    def test_next_cache_created_if_candle_window_exceeded(
        self, mock_get_max_timestamp_to
    ):
        """Next cache created, if candle window is exceeded."""
        filtered_1 = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, filtered_1)
        filtered_2 = self.get_filtered(self.one_minute_from_now)
        self.write_trade_data(
            self.one_minute_from_now, self.two_minutes_from_now, filtered_2
        )
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all()
        self.assertFalse(candle_data.exists())
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)

    def test_one_candle_from_one_trade_in_the_first_minute_and_another_in_the_second(
        self, mock_get_max_timestamp_to
    ):
        """
        One candle from one trade in the first minute, and another in the second.
        """
        filtered_1 = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, filtered_1)
        filtered_2 = self.get_filtered(self.one_minute_from_now)
        self.write_trade_data(
            self.one_minute_from_now, self.two_minutes_from_now, filtered_2
        )
        for i in range(2):
            aggregate_candles(
                self.candle,
                self.timestamp_from + pd.Timedelta(f"{i}t"),
                self.one_minute_from_now + pd.Timedelta(f"{i}t"),
            )

        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 2)
        self.assertGreater(len(candle_cache.first().json_data), 0)
        self.assertEqual(len(candle_cache.last().json_data), 0)

        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        candle_data = candle_data.first()
        self.assertEqual(candle_data.timestamp, self.timestamp_from)

        df = pd.concat([filtered_1, filtered_2])
        candle = aggregate_candle(df)
        del candle["timestamp"]
        self.assertEqual(candle_data.json_data, candle)


@override_settings(IS_LOCAL=False)
@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=timezone.utc),
)
class TimeBasedHourFrequencyCandleTest(
    BaseHourIteratorTest, BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):
    databases = {"default", "read_only"}

    def get_candle(self) -> Candle:
        """Get candle."""
        return TimeBasedCandle.objects.create(json_data={"window": "1h"})

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        """Write trade data."""
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, validated={}
        )

    def test_one_candle_from_trade_in_the_first_hour(self, mock_get_max_timestamp_to):
        """One candle from a trade in the first hour."""
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_one_candle_from_trade_in_the_first_hour_with_retry(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade in the first hour with retry."""
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        for i in range(2):
            aggregate_candles(
                self.candle,
                self.timestamp_from,
                self.one_hour_from_now,
                retry=bool(i),
            )
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    @override_settings(IS_LOCAL=True)
    def test_one_candle_from_trade_in_the_first_hour_read_only(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade in the first hour, read only."""
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleReadOnlyData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_two_candles_from_trades_in_the_first_and_second_hour(
        self, mock_get_max_timestamp_to
    ):
        """Two candles, one in the first hour, and another in the second hour."""
        filtered_1 = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        filtered_2 = self.get_filtered(self.one_hour_from_now)
        self.write_trade_data(
            self.one_hour_from_now, self.two_hours_from_now, filtered_2
        )
        aggregate_candles(self.candle, self.timestamp_from, self.two_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        self.assertEqual(candle_data[1].timestamp, self.one_hour_from_now)

    def test_two_candles_from_trades_in_first_and_third_hour(
        self, mock_get_max_timestamp_to
    ):
        """
        Two candles, one in the first hour, and another in the third hour.
        No trades, no candle, for the second hour.
        """
        filtered_1 = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.one_hour_from_now,
            frequency=Frequency.HOUR,
        )
        filtered_2 = self.get_filtered(self.two_hours_from_now)
        self.write_trade_data(
            self.two_hours_from_now, self.three_hours_from_now, filtered_2
        )
        aggregate_candles(self.candle, self.timestamp_from, self.three_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        self.assertEqual(candle_data[1].timestamp, self.two_hours_from_now)

    def test_candle_cache_created_from_trade_in_the_first_minute(
        self, mock_get_max_timestamp_to
    ):
        """Candle cache created from a trade in the first minute."""
        filtered = self.get_filtered(self.timestamp_from)
        one_minute_from_now = self.timestamp_from + pd.Timedelta("1t")
        self.write_trade_data(self.timestamp_from, one_minute_from_now, filtered)
        aggregate_candles(self.candle, self.timestamp_from, one_minute_from_now)
        candle_data = CandleData.objects.all()
        self.assertFalse(candle_data.exists())
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)
        candle = aggregate_candle(filtered)
        self.assertEqual(candle_cache.first().json_data["next"], candle)

    def test_one_candle_from_trade_with_existing_one_minute_candle_cache(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade, with existing one minute candle cache."""
        filtered_1 = self.get_filtered(self.timestamp_from)
        CandleCache.objects.create(
            candle=self.candle,
            timestamp=self.timestamp_from,
            frequency=Frequency.MINUTE,
            json_data=get_next_cache(filtered_1, {}),
        )
        one_minute_from_now = self.timestamp_from + pd.Timedelta("1t")
        filtered_2 = self.get_filtered(one_minute_from_now)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_2)
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        last_candle_cache = CandleCache.objects.last()
        self.assertEqual(last_candle_cache.json_data, {})


@override_settings(IS_LOCAL=False)
@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=timezone.utc),
)
class TimeBasedTwoHourFrequencyCandleTest(
    BaseHourIteratorTest, BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):
    databases = {"default", "read_only"}

    def get_candle(self) -> Candle:
        """Get candle."""
        return TimeBasedCandle.objects.create(json_data={"window": "2h"})

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        """Write trade data."""
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, validated={}
        )

    def test_next_cache_created_if_candle_window_exceeded(
        self, mock_get_max_timestamp_to
    ):
        """Next cache created, if candle window is exceeded."""
        filtered_1 = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        filtered_2 = self.get_filtered(self.one_hour_from_now)
        self.write_trade_data(
            self.one_hour_from_now, self.two_hours_from_now, filtered_2
        )
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertFalse(candle_data.exists())
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)

    def test_one_candle_from_one_trade_in_the_first_hour_and_another_in_the_second(
        self, mock_get_max_timestamp_to
    ):
        """One candle from one trade in the first hour, and another in the second."""
        filtered_1 = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        filtered_2 = self.get_filtered(self.one_hour_from_now)
        self.write_trade_data(
            self.one_hour_from_now, self.two_hours_from_now, filtered_2
        )
        for i in range(2):
            aggregate_candles(
                self.candle,
                self.timestamp_from + pd.Timedelta(f"{i}h"),
                self.one_hour_from_now + pd.Timedelta(f"{i}h"),
            )

        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 2)
        self.assertGreater(len(candle_cache.first().json_data), 0)
        self.assertEqual(len(candle_cache.last().json_data), 0)

        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        candle_data = candle_data.first()
        self.assertEqual(candle_data.timestamp, self.timestamp_from)

        df = pd.concat([filtered_1, filtered_2])
        candle = aggregate_candle(df)
        del candle["timestamp"]
        self.assertEqual(candle_data.json_data, candle)


@override_settings(IS_LOCAL=False)
@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=timezone.utc),
)
class ConstantNotionalHourFrequencyCandleTest(
    BaseHourIteratorTest, BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):
    databases = {"default", "read_only"}

    def get_candle(self) -> Candle:
        """Get candle."""
        return ConstantCandle.objects.create(
            json_data={"sample_type": SampleType.NOTIONAL, "target_value": 1}
        )

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        """Write trade data."""
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, validated={}
        )

    def test_no_candles_from_trade_in_the_first_hour(self, mock_get_max_timestamp_to):
        """No candles with one trade with insufficent notional in the first hour."""
        expected = Decimal("0.5")
        filtered = self.get_filtered(self.timestamp_from, notional=expected)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        self.assertFalse(CandleData.objects.exists())
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)
        self.assertEqual(candle_cache[0].json_data["sample_value"], expected)

    def test_one_candle_from_trade_in_the_first_hour(self, mock_get_max_timestamp_to):
        """One candle from a trade in the first hour."""
        filtered = self.get_filtered(self.timestamp_from, notional=1)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_one_candle_from_trade_in_the_first_hour_with_retry(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade in the first hour with retry."""
        filtered = self.get_filtered(self.timestamp_from, notional=1)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        for i in range(2):
            aggregate_candles(
                self.candle,
                self.timestamp_from,
                self.one_hour_from_now,
                retry=bool(i),
            )
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_one_candle_from_one_trade_in_the_first_hour_then_two_trades_with_retry(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade in the first hour, then two trades with retry."""
        filtered = self.get_filtered(self.timestamp_from, notional=Decimal("1"))
        for i in range(2):
            retry = bool(i)
            if retry:
                missing_trade = self.get_filtered(
                    self.timestamp_from, notional=Decimal("0.5")
                )
                filtered = pd.concat([filtered, missing_trade])
            self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
            aggregate_candles(
                self.candle, self.timestamp_from, self.one_hour_from_now, retry=True
            )
        querysets = {
            TradeData: TradeData.objects.all(),
            CandleCache: CandleCache.objects.all(),
            CandleData: CandleData.objects.all(),
        }
        for model, queryset in querysets.items():
            self.assertEqual(queryset.count(), 1)
        self.assertEqual(querysets[CandleData][0].timestamp, self.timestamp_from)

    @override_settings(IS_LOCAL=True)
    def test_one_candle_from_trade_in_the_first_hour_read_only(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade in the first hour, read only."""
        filtered = self.get_filtered(self.timestamp_from, notional=1)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleReadOnlyData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_one_candle_from_two_trades_in_the_first_hour_with_topN(
        self, mock_get_max_timestamp_to
    ):
        """One candle from two trades in the first hour with topN."""
        filtered_1 = self.get_filtered(self.timestamp_from, notional=Decimal("0.25"))
        filtered_2 = self.get_filtered(
            self.timestamp_from, nanoseconds=1, notional=Decimal("0.75")
        )
        filtered = pd.concat([filtered_1, filtered_2])
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        self.candle.json_data["topN"] = 1
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        top_n = candle_data[0].json_data["topN"]
        self.assertEqual(len(top_n), 1)
        self.assertEqual(top_n[0]["timestamp"], filtered_2.iloc[0].timestamp)
        self.assertEqual(top_n[0]["nanoseconds"], filtered_2.iloc[0].nanoseconds)

    def test_one_candle_from_trade_in_the_first_and_second_hour(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade in the first and second hour."""
        filtered_1 = self.get_filtered(self.timestamp_from, notional=Decimal("0.5"))
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        filtered_2 = self.get_filtered(self.one_hour_from_now, notional=Decimal("0.5"))
        self.write_trade_data(
            self.one_hour_from_now, self.two_hours_from_now, filtered_2
        )
        aggregate_candles(self.candle, self.one_hour_from_now, self.two_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        data_frame = self.candle.get_data_frame(
            self.timestamp_from, self.two_hours_from_now
        )
        data = candle_data[0].json_data
        self.assertEqual(data["notional"], data_frame["totalNotional"].sum())
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 2)
        self.assertIn("next", candle_cache[0].json_data)
        self.assertNotIn("next", candle_cache[1].json_data)

    def test_one_candle_from_trade_in_the_first_and_second_hour_with_topN(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade in the first and second hour, with topN."""
        filtered_1 = self.get_filtered(self.timestamp_from, notional=Decimal("0.5"))
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        filtered_2 = self.get_filtered(self.one_hour_from_now, notional=Decimal("0.75"))
        self.write_trade_data(
            self.one_hour_from_now, self.two_hours_from_now, filtered_2
        )
        self.candle.json_data["topN"] = 1
        aggregate_candles(self.candle, self.one_hour_from_now, self.two_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        data_frame = self.candle.get_data_frame(
            self.timestamp_from, self.two_hours_from_now
        )
        data = candle_data[0].json_data
        self.assertEqual(data["notional"], data_frame["totalNotional"].sum())
        topN = data["topN"]
        self.assertEqual(len(topN), 1)
        self.assertEqual(topN[0]["timestamp"], filtered_2.iloc[0].timestamp)
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 2)
        self.assertIn("next", candle_cache[0].json_data)
        self.assertNotIn("next", candle_cache[1].json_data)

    def test_two_candles_from_trades_in_the_first_and_second_hour(
        self, mock_get_max_timestamp_to
    ):
        """Two candles, one in the first hour, and another in the second hour."""
        filtered_1 = self.get_filtered(self.timestamp_from, notional=1)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        filtered_2 = self.get_filtered(self.one_hour_from_now, notional=1)
        self.write_trade_data(
            self.one_hour_from_now, self.two_hours_from_now, filtered_2
        )
        aggregate_candles(self.candle, self.timestamp_from, self.two_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        self.assertEqual(candle_data[1].timestamp, self.one_hour_from_now)

    def test_two_candles_from_trades_in_first_and_third_hour(
        self, mock_get_max_timestamp_to
    ):
        """
        Two candles, one in the first hour, and another in the third hour.
        No trades, no candle, in the second hour.
        """
        self.write_trade_data(
            self.timestamp_from,
            self.one_hour_from_now,
            self.get_filtered(self.timestamp_from, notional=1),
        )
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.one_hour_from_now,
            frequency=Frequency.HOUR,
        )
        self.write_trade_data(
            self.two_hours_from_now,
            self.three_hours_from_now,
            self.get_filtered(self.two_hours_from_now, notional=1),
        )
        aggregate_candles(self.candle, self.timestamp_from, self.three_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        self.assertEqual(candle_data[1].timestamp, self.two_hours_from_now)

    def test_no_candles_without_prior_cache(self, mock_get_max_timestamp_to):
        """No candles without prior cache."""
        CandleCache.objects.create(
            candle=self.candle, timestamp=self.timestamp_from, frequency=Frequency.HOUR
        )
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.two_hours_from_now,
            frequency=Frequency.HOUR,
        )
        aggregate_candles(
            self.candle, self.two_hours_from_now, self.three_hours_from_now
        )
        self.assertFalse(CandleData.objects.exists())
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)


@override_settings(IS_LOCAL=False)
@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 5).replace(tzinfo=timezone.utc),
)
class ConstantNotionalDayFrequencyIrregularCandleTest(
    BaseDayIteratorTest, BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):
    databases = {"default", "read_only"}

    def get_candle(self) -> Candle:
        """Get candle."""
        return ConstantCandle.objects.create(
            json_data={
                "sample_type": SampleType.NOTIONAL,
                "target_value": 1,
                "cache_reset": Frequency.DAY,
            }
        )

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        """Write trade data."""
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, validated={}
        )

    def test_one_incomplete_candle(self, mock_get_max_timestamp_to):
        """An incomplete candle is saved, if cache will be reset."""
        last_hour = self.timestamp_from + pd.Timedelta("23h")
        filtered = self.get_filtered(last_hour, notional=Decimal("0.5"))
        self.write_trade_data(last_hour, self.one_day_from_now, filtered)
        aggregate_candles(self.candle, last_hour, self.one_day_from_now)
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, last_hour)
        self.assertTrue(candle_data[0].json_data["incomplete"])

    def test_one_candle_from_trade_with_existing_one_minute_candle_cache(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade, with existing one minute candle cache."""
        CandleCache.objects.create(
            candle=self.candle,
            timestamp=self.timestamp_from,
            frequency=Frequency.MINUTE,
            json_data={"sample_value": 0},
        )
        one_minute_from_now = self.timestamp_from + pd.Timedelta("1t")
        one_hour_from_now = self.timestamp_from + pd.Timedelta("1h")
        filtered_1 = self.get_filtered(one_minute_from_now)
        self.write_trade_data(self.timestamp_from, one_hour_from_now, filtered_1)
        filtered_2 = self.get_filtered(one_minute_from_now)
        self.write_trade_data(self.timestamp_from, one_hour_from_now, filtered_2)
        aggregate_candles(self.candle, self.timestamp_from, one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, one_minute_from_now)
        candle_cache = CandleCache.objects.last()
        self.assertNotIn("next", candle_cache.json_data)


@override_settings(IS_LOCAL=False)
@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=timezone.utc),
)
class AdaptiveNotionalCandleTest(
    BaseHourIteratorTest, BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):
    databases = {"default", "read_only"}

    def setUp(self):
        super().setUp()
        self.create_trade_data_summary(self.timestamp_from)

    def get_candle(self) -> Candle:
        """Get candle."""
        return AdaptiveCandle.objects.create(
            json_data={
                "sample_type": SampleType.NOTIONAL,
                "moving_average_number_of_days": 1,
                "target_candles_per_day": 1,
            }
        )

    def create_trade_data_summary(
        self, timestamp: datetime, notional: Decimal = Decimal("1")
    ) -> None:
        """Create trade data summary."""
        ts = get_min_time(timestamp, value="1d")
        one_day_ago = ts - pd.Timedelta("1d")
        filtered = self.get_filtered(one_day_ago, notional=notional)
        trade_data = TradeData.objects.create(
            symbol=self.symbol, timestamp=one_day_ago, frequency=Frequency.DAY
        )
        TradeData.write_data_frame(trade_data, filtered, pd.DataFrame([]))
        for symbol in Symbol.objects.all():
            dates = (
                TradeData.objects.filter(symbol=symbol)
                .values("timestamp")
                .annotate(date=TruncDate("timestamp"))
                .values_list("date", flat=True)
                .distinct()
                # Not distinct without order_by.
                .order_by()
            )
            for date in dates:
                TradeDataSummary.aggregate(symbol, date)

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        """Write trade data."""
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, validated={}
        )

    def test_no_candles_from_trade_in_the_first_hour(self, mock_get_max_timestamp_to):
        """No candles with one trade with insufficent notional in the first hour."""
        expected = Decimal("0.5")
        filtered = self.get_filtered(self.timestamp_from, notional=expected)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        self.assertFalse(CandleData.objects.exists())
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)
        self.assertEqual(candle_cache[0].json_data["sample_value"], expected)

    def test_one_candle_from_trade_in_the_first_hour(self, mock_get_max_timestamp_to):
        """One candle from a trade in the first hour."""
        filtered = self.get_filtered(self.timestamp_from, notional=1)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_one_candle_from_trade_in_the_first_hour_with_retry(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade in the first hour with retry."""
        filtered = self.get_filtered(self.timestamp_from, notional=1)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        for i in range(2):
            aggregate_candles(
                self.candle,
                self.timestamp_from,
                self.one_hour_from_now,
                retry=bool(i),
            )
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_one_candle_from_one_trade_in_the_first_hour_then_two_trades_with_retry(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade in the first hour, then two trades with retry."""
        filtered = self.get_filtered(self.timestamp_from, notional=Decimal("1"))
        for i in range(2):
            retry = bool(i)
            if retry:
                missing_trade = self.get_filtered(
                    self.timestamp_from, notional=Decimal("0.5")
                )
                filtered = pd.concat([filtered, missing_trade])
            self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
            aggregate_candles(
                self.candle, self.timestamp_from, self.one_hour_from_now, retry=True
            )
        self.assertEqual(TradeData.objects.count(), 2)
        self.assertEqual(CandleCache.objects.count(), 1)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    @override_settings(IS_LOCAL=True)
    def test_one_candle_from_trade_in_the_first_hour_read_only(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade in the first hour, read only."""
        filtered = self.get_filtered(self.timestamp_from, notional=1)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleReadOnlyData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_one_candle_from_two_trades_in_the_first_hour_with_topN(
        self, mock_get_max_timestamp_to
    ):
        """One candle from two trades in the first hour with topN."""
        filtered_1 = self.get_filtered(self.timestamp_from, notional=Decimal("0.25"))
        filtered_2 = self.get_filtered(
            self.timestamp_from, nanoseconds=1, notional=Decimal("0.75")
        )
        filtered = pd.concat([filtered_1, filtered_2])
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        self.candle.json_data["topN"] = 1
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        top_n = candle_data[0].json_data["topN"]
        self.assertEqual(len(top_n), 1)
        self.assertEqual(top_n[0]["timestamp"], filtered_2.iloc[0].timestamp)
        self.assertEqual(top_n[0]["nanoseconds"], filtered_2.iloc[0].nanoseconds)

    def test_one_candle_from_trade_in_the_first_and_second_hour(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade in the first and second hour."""
        filtered_1 = self.get_filtered(self.timestamp_from, notional=Decimal("0.5"))
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        filtered_2 = self.get_filtered(self.one_hour_from_now, notional=Decimal("0.5"))
        self.write_trade_data(
            self.one_hour_from_now, self.two_hours_from_now, filtered_2
        )
        aggregate_candles(self.candle, self.one_hour_from_now, self.two_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        data_frame = self.candle.get_data_frame(
            self.timestamp_from, self.two_hours_from_now
        )
        data = candle_data[0].json_data
        self.assertEqual(data["notional"], data_frame["totalNotional"].sum())
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 2)
        self.assertIn("next", candle_cache[0].json_data)
        self.assertNotIn("next", candle_cache[1].json_data)

    def test_one_candle_from_trade_in_the_first_and_second_hour_with_topN(
        self, mock_get_max_timestamp_to
    ):
        """One candle from a trade in the first and second hour, with topN."""
        filtered_1 = self.get_filtered(self.timestamp_from, notional=Decimal("0.5"))
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        filtered_2 = self.get_filtered(self.one_hour_from_now, notional=Decimal("0.75"))
        self.write_trade_data(
            self.one_hour_from_now, self.two_hours_from_now, filtered_2
        )
        self.candle.json_data["topN"] = 1
        aggregate_candles(self.candle, self.one_hour_from_now, self.two_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        data_frame = self.candle.get_data_frame(
            self.timestamp_from, self.two_hours_from_now
        )
        data = candle_data[0].json_data
        self.assertEqual(data["notional"], data_frame["totalNotional"].sum())
        topN = data["topN"]
        self.assertEqual(len(topN), 1)
        self.assertEqual(topN[0]["timestamp"], filtered_2.iloc[0].timestamp)
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 2)
        self.assertIn("next", candle_cache[0].json_data)
        self.assertNotIn("next", candle_cache[1].json_data)

    def test_two_candles_from_trades_in_the_first_and_second_hour(
        self, mock_get_max_timestamp_to
    ):
        """Two candles, one in the first hour, and another in the second hour."""
        filtered_1 = self.get_filtered(self.timestamp_from, notional=1)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        filtered_2 = self.get_filtered(self.one_hour_from_now, notional=1)
        self.write_trade_data(
            self.one_hour_from_now, self.two_hours_from_now, filtered_2
        )
        aggregate_candles(self.candle, self.timestamp_from, self.two_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        self.assertEqual(candle_data[1].timestamp, self.one_hour_from_now)

    def test_two_candles_from_trades_in_first_and_third_hour(
        self, mock_get_max_timestamp_to
    ):
        """
        Two candles, one in the first hour, and another in the third hour.
        No trades, no candle, in the second hour.
        """
        self.write_trade_data(
            self.timestamp_from,
            self.one_hour_from_now,
            self.get_filtered(self.timestamp_from, notional=1),
        )
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.one_hour_from_now,
            frequency=Frequency.HOUR,
        )
        self.write_trade_data(
            self.two_hours_from_now,
            self.three_hours_from_now,
            self.get_filtered(self.two_hours_from_now, notional=1),
        )
        aggregate_candles(self.candle, self.timestamp_from, self.three_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        self.assertEqual(candle_data[1].timestamp, self.two_hours_from_now)
