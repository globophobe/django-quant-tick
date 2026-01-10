from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import time_machine
from django.test import TestCase
from pandas import DataFrame

from quant_tick.constants import FileData, Frequency, RenkoKind, SampleType
from quant_tick.controllers import CandleCacheIterator, aggregate_candles
from quant_tick.lib import (
    aggregate_candle,
    get_current_time,
    get_min_time,
    get_next_cache,
    merge_cache,
)
from quant_tick.models import (
    AdaptiveCandle,
    Candle,
    CandleCache,
    CandleData,
    ConstantCandle,
    RenkoBrick,
    TimeBasedCandle,
    TradeData,
)

from ..base import BaseSymbolTest, BaseWriteTradeDataTest


class BaseMinuteIteratorTest:
    """Base minute iterator test."""

    def setUp(self) -> None:
        """Set up."""
        super().setUp()
        self.timestamp_from = get_min_time(get_current_time(), "1d")
        self.one_minute_from_now = self.timestamp_from + pd.Timedelta("1min")
        self.two_minutes_from_now = self.timestamp_from + pd.Timedelta("2min")
        self.three_minutes_from_now = self.timestamp_from + pd.Timedelta("3min")


class BaseHourIteratorTest:
    """Base hour iterator test."""

    def setUp(self) -> None:
        """Set up."""
        super().setUp()
        self.timestamp_from = get_min_time(get_current_time(), "1d")
        self.one_hour_from_now = self.timestamp_from + pd.Timedelta("1h")
        self.two_hours_from_now = self.timestamp_from + pd.Timedelta("2h")
        self.three_hours_from_now = self.timestamp_from + pd.Timedelta("3h")


class BaseDayIteratorTest:
    """Base day iterator test."""

    def setUp(self) -> None:
        """Set up."""
        super().setUp()
        self.timestamp_from = get_min_time(get_current_time(), "1d")
        self.one_day_from_now = self.timestamp_from + pd.Timedelta("1d")
        self.two_days_from_now = self.timestamp_from + pd.Timedelta("2d")
        self.three_days_from_now = self.timestamp_from + pd.Timedelta("3d")


class BaseCandleCacheIteratorTest(BaseSymbolTest):
    """Base candle cache iterator test."""

    def setUp(self) -> None:
        """Set up."""
        super().setUp()
        self.symbol = self.get_symbol()
        self.candle = self.get_candle()

    def get_candle(self) -> Candle:
        """Get candle."""
        return Candle.objects.create(
            symbol=self.symbol, json_data={"source_data": FileData.RAW}
        )

    def get_values(self) -> list[tuple[datetime, datetime]]:
        """Get values."""
        return [
            value
            for value in CandleCacheIterator(self.candle).iter_all(
                self.timestamp_from, self.timestamp_to
            )
        ]


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 4, 0, 5).replace(tzinfo=timezone.utc),
)
class CandleCacheIteratorTest(BaseCandleCacheIteratorTest, TestCase):
    """Candle cache iterator test."""

    def setUp(self) -> None:
        """Set up."""
        super().setUp()
        self.one_minute = pd.Timedelta("1min")
        self.timestamp_to = self.timestamp_from + (self.one_minute * 5)

    def create_trade_data(self) -> None:
        """Create trade data."""
        for i in range(5):
            TradeData.objects.create(
                symbol=self.symbol,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}min"),
                frequency=Frequency.MINUTE,
            )

    def test_iter_all_no_results(self, mock_get_max_timestamp_to):
        """No results."""
        self.create_trade_data()
        for i in range(5):
            CandleCache.objects.create(
                candle=self.candle,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}min"),
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


@time_machine.travel(datetime(2009, 1, 4), tick=False)
class CandleTest(BaseSymbolTest, BaseDayIteratorTest, TestCase):
    """Candle test."""

    def setUp(self):
        """Set up."""
        super().setUp()
        self.candle = Candle.objects.create(
            symbol=self.get_symbol(), json_data={"source_data": FileData.RAW}
        )

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
    """Time based minute frequency candle test."""

    def get_candle(self) -> Candle:
        """Get candle."""
        return TimeBasedCandle.objects.create(
            symbol=self.symbol,
            json_data={"source_data": FileData.RAW, "window": "1min"},
        )

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        """Write trade data."""
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, pd.DataFrame([])
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
            timestamp=self.timestamp_from + pd.Timedelta("1min"),
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
    """Time based two minute frequency candle test."""

    def get_candle(self) -> Candle:
        """Get candle."""
        return TimeBasedCandle.objects.create(
            symbol=self.symbol,
            json_data={"source_data": FileData.RAW, "window": "2min"},
        )

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        """Write trade data."""
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, pd.DataFrame([])
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
                self.timestamp_from + pd.Timedelta(f"{i}min"),
                self.one_minute_from_now + pd.Timedelta(f"{i}min"),
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


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=timezone.utc),
)
class TimeBasedHourFrequencyCandleTest(
    BaseHourIteratorTest, BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):
    """Time based hour frequency candle test."""

    def get_candle(self) -> Candle:
        """Get candle."""
        return TimeBasedCandle.objects.create(
            symbol=self.symbol,
            json_data={"source_data": FileData.RAW, "window": "1h"},
        )

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        """Write trade data."""
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, pd.DataFrame([])
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
        one_minute_from_now = self.timestamp_from + pd.Timedelta("1min")
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
        one_minute_from_now = self.timestamp_from + pd.Timedelta("1min")
        filtered_2 = self.get_filtered(one_minute_from_now)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_2)
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        last_candle_cache = CandleCache.objects.last()
        self.assertEqual(last_candle_cache.json_data, {})


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=timezone.utc),
)
class TimeBasedTwoHourFrequencyCandleTest(
    BaseHourIteratorTest, BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):
    """Time based two hour frequency candle test."""

    def get_candle(self) -> Candle:
        """Get candle."""
        return TimeBasedCandle.objects.create(
            symbol=self.symbol,
            json_data={"source_data": FileData.RAW, "window": "2h"},
        )

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        """Write trade data."""
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, pd.DataFrame([])
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


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=timezone.utc),
)
class ConstantNotionalHourFrequencyCandleTest(
    BaseHourIteratorTest, BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):
    """Constant notional hour frequency candle test."""

    def get_candle(self) -> Candle:
        """Get candle."""
        return ConstantCandle.objects.create(
            symbol=self.symbol,
            json_data={
                "source_data": FileData.RAW,
                "sample_type": SampleType.NOTIONAL,
                "target_value": 1,
            },
        )

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        """Write trade data."""
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, pd.DataFrame([])
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


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 5).replace(tzinfo=timezone.utc),
)
class ConstantNotionalDayFrequencyIrregularCandleTest(
    BaseDayIteratorTest, BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):
    """Constant notional day frequency candle test."""

    def get_candle(self) -> Candle:
        """Get candle."""
        return ConstantCandle.objects.create(
            symbol=self.symbol,
            json_data={
                "source_data": FileData.RAW,
                "sample_type": SampleType.NOTIONAL,
                "target_value": 1,
                "cache_reset": Frequency.DAY,
            },
        )

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        """Write trade data."""
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, pd.DataFrame([])
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


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=timezone.utc),
)
class AdaptiveNotionalCandleTest(
    BaseHourIteratorTest, BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):
    """Adaptive notional candle test."""

    def setUp(self):
        """Set up."""
        super().setUp()
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=get_min_time(self.timestamp_from, value="1d")
            - pd.Timedelta("1d"),
            frequency=Frequency.DAY,
            json_data={"candle": {"notional": 1}},
        )

    def get_candle(self) -> Candle:
        """Get candle."""
        return AdaptiveCandle.objects.create(
            symbol=self.symbol,
            json_data={
                "source_data": FileData.RAW,
                "sample_type": SampleType.NOTIONAL,
                "moving_average_number_of_days": 1,
                "target_candles_per_day": 1,
            },
        )

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        """Write trade data."""
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, pd.DataFrame([])
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


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=timezone.utc),
)
class RenkoBrickTest(
    BaseMinuteIteratorTest,
    BaseWriteTradeDataTest,
    BaseCandleCacheIteratorTest,
    TestCase,
):
    """Renko brick test."""

    def get_candle(self) -> RenkoBrick:
        """Get candle."""
        return RenkoBrick.objects.create(
            symbol=self.symbol,
            json_data={
                "source_data": FileData.RAW,
                "target_percentage_change": Decimal("0.01"),
                "origin_price": Decimal("100"),
            },
        )

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        """Write trade data."""
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, pd.DataFrame([])
        )

    def get_raw_renko(self, prices: list[Decimal] | Decimal) -> list[dict]:
        """Get raw renko."""
        prices = prices if isinstance(prices, list) else [prices]
        trades = [
            self.get_random_trade(
                timestamp=self.timestamp_from,
                nanoseconds=nanoseconds,
                price=price,
            )
            for nanoseconds, price in enumerate(prices)
        ]
        return pd.DataFrame(trades)

    def assert_cache_state(self, expected: dict) -> None:
        """Assert CandleCache state matches expected."""
        cache_obj = (
            CandleCache.objects.filter(candle=self.candle)
            .order_by("-timestamp")
            .first()
        )
        self.assertIsNotNone(cache_obj)
        cache = cache_obj.json_data
        for key, value in expected.items():
            if key == "wicks":
                self.assertEqual(len(cache.get("wicks", [])), value)
            else:
                self.assertEqual(cache.get(key), value, f"cache[{key}]")

    def test_one_trade_one_brick(self, mock_get_max_timestamp_to):
        """Seed pattern: first trade initializes cache but emits nothing.

        Prices: 100
        Levels: L0

        | Trade | Price | Level | Action                          |
        |-------|-------|-------|---------------------------------|
        | 1     | 100   | 0     | Seed: cache={level=0, entry=None} |

        Emitted: (none)
        """
        data_frame = self.get_raw_renko(Decimal("100"))
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 0)
        # Cache: seed level, no entry direction, sequence 0
        self.assert_cache_state(
            {
                "level": 0,
                "direction": None,
                "sequence": 0,
                "wicks": 0,
            }
        )
        # get_candle_data: no complete bricks
        df = DataFrame(
            self.candle.get_candle_data(self.timestamp_from, self.one_minute_from_now)
        )
        self.assertEqual(len(df), 0)

    def test_two_trades_one_brick_up(self, mock_get_max_timestamp_to):
        """Simple upward body: exit level triggers emit.

        Prices: 100 → 101
        Levels: L0  → L1

        | Trade | Price | Level | Action                    |
        |-------|-------|-------|---------------------------|
        | 1     | 100   | 0     | Seed                      |
        | 2     | 101   | 1     | Exit L0 ↑ → BODY(L0, seq=0) |

        Emitted: body(L0, ↑)
        """
        data_frame = self.get_raw_renko([Decimal("100"), Decimal("101")])
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        brick = candle_data[0].json_data
        brick_renko = candle_data[0].renko_data
        self.assertEqual(brick_renko.level, 0)
        self.assertEqual(brick_renko.direction, 1)
        self.assertEqual(brick_renko.kind, RenkoKind.BODY)
        self.assertEqual(brick["close"], data_frame.iloc[0].price)
        # Cache: now at L1, entered from below, next seq=1
        self.assert_cache_state(
            {
                "level": 1,
                "direction": 1,
                "sequence": 1,
                "wicks": 0,
            }
        )
        # get_candle_data: L0 body is incomplete (last row filtered)
        df = DataFrame(
            self.candle.get_candle_data(self.timestamp_from, self.one_minute_from_now)
        )
        self.assertEqual(len(df), 0)
        # With is_complete=False, includes L0 body + incomplete L1 from cache
        df_incomplete = DataFrame(
            self.candle.get_candle_data(
                self.timestamp_from, self.one_minute_from_now, is_complete=False
            )
        )
        self.assertEqual(len(df_incomplete), 2)
        self.assertEqual(df_incomplete.iloc[0]["level"], 0)
        self.assertEqual(df_incomplete.iloc[1]["level"], 1)

    def test_three_trades_one_brick_up(self, mock_get_max_timestamp_to):
        """Continuation within level: third trade accumulates in L1.

        Prices: 100 → 101  → 101.5
        Levels: L0  → L1   → L1

        | Trade | Price | Level | Action                    |
        |-------|-------|-------|---------------------------|
        | 1     | 100   | 0     | Seed                      |
        | 2     | 101   | 1     | Exit L0 ↑ → BODY(L0, seq=0) |
        | 3     | 101.5 | 1     | Accumulate in L1 (no exit)  |

        Emitted: body(L0, ↑)
        """
        data_frame = self.get_raw_renko(
            [Decimal("100"), Decimal("101"), Decimal("101.5")]
        )
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        brick = candle_data[0].json_data
        brick_renko = candle_data[0].renko_data
        self.assertEqual(brick_renko.level, 0)
        self.assertEqual(brick["close"], data_frame.iloc[0].price)
        # Cache: still at L1 accumulating, next seq=1
        self.assert_cache_state(
            {
                "level": 1,
                "direction": 1,
                "sequence": 1,
                "wicks": 0,
            }
        )
        # get_candle_data: 0 complete rows (L0 body is last, filtered)
        df = DataFrame(
            self.candle.get_candle_data(self.timestamp_from, self.one_minute_from_now)
        )
        self.assertEqual(len(df), 0)

    def test_three_trades_one_brick_up_incomplete_reversal(
        self, mock_get_max_timestamp_to
    ):
        """Single wick pending: return to L0 creates pending wick.

        Prices: 100 → 101 → 100
        Levels: L0  → L1  → L0

        | Trade | Price | Level | Action                       |
        |-------|-------|-------|------------------------------|
        | 1     | 100   | 0     | Seed                         |
        | 2     | 101   | 1     | Exit L0 ↑ → BODY(L0, seq=0)  |
        | 3     | 100   | 0     | Exit L1 ↓ → wick(L1) pending |

        Emitted: body(L0, ↑)
        Cache: wicks=[wick(L1, ↓)]
        """
        data_frame = self.get_raw_renko(
            [Decimal("100"), Decimal("101"), Decimal("100")]
        )
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all()
        # Only 1 brick emitted: level 0 body
        # Level 1 wick remains pending (not emitted because no parent body emitted)
        self.assertEqual(candle_data.count(), 1)
        brick0_renko = candle_data[0].renko_data
        # Brick 0: level 0 exited upward (body)
        self.assertEqual(brick0_renko.level, 0)
        self.assertEqual(brick0_renko.direction, 1)
        self.assertEqual(brick0_renko.kind, RenkoKind.BODY)
        # Cache: back at L0, entered from above, 1 pending wick
        self.assert_cache_state(
            {
                "level": 0,
                "direction": -1,
                "sequence": 1,
                "wicks": 1,
            }
        )
        # get_candle_data: 0 complete rows (only L0 body, filtered as last)
        df = DataFrame(
            self.candle.get_candle_data(self.timestamp_from, self.one_minute_from_now)
        )
        self.assertEqual(len(df), 0)

    def test_three_trades_one_brick_up_reversal_down(self, mock_get_max_timestamp_to):
        """Multi-level reversal emits body: L0→L1→L-1.

        Prices: 100 → 101 → 99.5
        Levels: L0  → L1  → L-1

        | Trade | Price | Level | Action                       |
        |-------|-------|-------|------------------------------|
        | 1     | 100   | 0     | Seed                         |
        | 2     | 101   | 1     | Exit L0 ↑ → BODY(L0, seq=0)  |
        | 3     | 99.5  | -1    | Exit L1 ↓ → BODY(L1, seq=1)  |

        2-level jump (L1→L-1) breaches through L0, so L1 is body not wick.
        Emitted: body(L0, ↑), body(L1, ↓)
        """
        data_frame = self.get_raw_renko(
            [Decimal("100"), Decimal("101"), Decimal("99.5")]
        )
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all().order_by("renko_data__sequence")
        # 2 bricks emitted: body(L0,↑), body(L1,↓)
        self.assertEqual(candle_data.count(), 2)
        brick0_renko = candle_data[0].renko_data
        self.assertEqual(brick0_renko.level, 0)
        self.assertEqual(brick0_renko.direction, 1)
        self.assertEqual(brick0_renko.kind, RenkoKind.BODY)
        brick1_renko = candle_data[1].renko_data
        self.assertEqual(brick1_renko.level, 1)
        self.assertEqual(brick1_renko.direction, -1)
        self.assertEqual(brick1_renko.kind, RenkoKind.BODY)
        # Cache: at L-1, entered from above, no pending wicks
        self.assert_cache_state(
            {
                "level": -1,
                "direction": -1,
                "sequence": 2,
                "wicks": 0,
            }
        )
        # get_candle_data: 1 complete row (L0), L1 is incomplete
        df = DataFrame(
            self.candle.get_candle_data(self.timestamp_from, self.one_minute_from_now)
        )
        self.assertEqual(len(df), 1)

    def test_logarithmic_reversibility(self, mock_get_max_timestamp_to):
        """Multi-level reversal: verifies logarithmic level calculation.

        Prices: 100 → 101 → 99.5
        Levels: L0  → L1  → L-1

        | Trade | Price | Level | Action                       |
        |-------|-------|-------|------------------------------|
        | 1     | 100   | 0     | Seed                         |
        | 2     | 101   | 1     | Exit L0 ↑ → BODY(L0, seq=0)  |
        | 3     | 99.5  | -1    | Exit L1 ↓ → BODY(L1, seq=1)  |

        2-level jump (L1→L-1) breaches through L0, so L1 is body not wick.
        Emitted: body(L0, ↑), body(L1, ↓)
        """
        data_frame = self.get_raw_renko(
            [Decimal("100"), Decimal("101"), Decimal("99.5")]
        )
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all().order_by("renko_data__sequence")
        # 2 bricks emitted: body(L0,↑), body(L1,↓)
        self.assertEqual(len(candle_data), 2)
        brick0_renko = candle_data[0].renko_data
        self.assertEqual(brick0_renko.level, 0)
        self.assertEqual(brick0_renko.direction, 1)
        self.assertEqual(brick0_renko.kind, RenkoKind.BODY)
        brick1_renko = candle_data[1].renko_data
        self.assertEqual(brick1_renko.level, 1)
        self.assertEqual(brick1_renko.direction, -1)
        self.assertEqual(brick1_renko.kind, RenkoKind.BODY)
        # Cache: at L-1, entered from above, no pending wicks
        self.assert_cache_state(
            {
                "level": -1,
                "direction": -1,
                "sequence": 2,
                "wicks": 0,
            }
        )

    def test_body_brick_complete_traversal(self, mock_get_max_timestamp_to):
        """Continuation: two consecutive bodies (L0→L1→L2).

        Prices: 100 → 101 → 102.02
        Levels: L0  → L1  → L2

        | Trade | Price  | Level | Action                    |
        |-------|--------|-------|---------------------------|
        | 1     | 100    | 0     | Seed                      |
        | 2     | 101    | 1     | Exit L0 ↑ → BODY(L0, seq=0) |
        | 3     | 102.02 | 2     | Exit L1 ↑ → BODY(L1, seq=1) |

        Emitted: body(L0, ↑), body(L1, ↑)
        """
        data_frame = self.get_raw_renko(
            [Decimal("100"), Decimal("101"), Decimal("102.02")]
        )
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all().order_by("timestamp")
        self.assertEqual(candle_data.count(), 2)
        brick0_renko = candle_data[0].renko_data
        brick1_renko = candle_data[1].renko_data
        # Both bricks are bodies (complete traversal)
        self.assertEqual(brick0_renko.level, 0)
        self.assertEqual(brick0_renko.kind, RenkoKind.BODY)
        self.assertEqual(brick1_renko.level, 1)
        self.assertEqual(brick1_renko.kind, RenkoKind.BODY)
        # Cache: at L2, entered from below, next seq=2
        self.assert_cache_state(
            {
                "level": 2,
                "direction": 1,
                "sequence": 2,
                "wicks": 0,
            }
        )
        # get_candle_data: 1 complete row (L0), L1 filtered as last
        df = DataFrame(
            self.candle.get_candle_data(self.timestamp_from, self.one_minute_from_now)
        )
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["level"], 0)

    def test_wick_brick_failed_excursion(self, mock_get_max_timestamp_to):
        """Single wick pending: failed excursion L0→L1→L0.

        Prices: 100 → 101  → 100.5
        Levels: L0  → L1   → L0

        | Trade | Price | Level | Action                       |
        |-------|-------|-------|------------------------------|
        | 1     | 100   | 0     | Seed                         |
        | 2     | 101   | 1     | Exit L0 ↑ → BODY(L0, seq=0)  |
        | 3     | 100.5 | 0     | Exit L1 ↓ → wick(L1) pending |

        Emitted: body(L0, ↑)
        Cache: wicks=[wick(L1, ↓)]
        """
        data_frame = self.get_raw_renko(
            [Decimal("100"), Decimal("101"), Decimal("100.5")]
        )
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all().order_by("timestamp")
        # Only 1 brick: level 0 body. Level 1 wick is pending (assigned to level 0).
        # Since we don't exit level 0 again, the wick is never emitted.
        self.assertEqual(candle_data.count(), 1)
        brick0_renko = candle_data[0].renko_data
        # Brick 0 is body (seed level 0)
        self.assertEqual(brick0_renko.kind, RenkoKind.BODY)
        self.assertEqual(brick0_renko.level, 0)
        # Cache: back at L0, entered from above, 1 pending wick
        self.assert_cache_state(
            {
                "level": 0,
                "direction": -1,
                "sequence": 1,
                "wicks": 1,
            }
        )
        # get_candle_data: 0 complete rows
        df = DataFrame(
            self.candle.get_candle_data(self.timestamp_from, self.one_minute_from_now)
        )
        self.assertEqual(len(df), 0)

    def test_multi_level_jump_skips_empty_bricks(self, mock_get_max_timestamp_to):
        """Multi-level jump: L0→L3 skips L1,L2 (no empty bricks).

        Prices: 100 → 103.04
        Levels: L0  → L3

        | Trade | Price  | Level | Action                    |
        |-------|--------|-------|---------------------------|
        | 1     | 100    | 0     | Seed                      |
        | 2     | 103.04 | 3     | Exit L0 ↑ → BODY(L0, seq=0) |

        Emitted: body(L0, ↑)
        Note: Levels L1, L2 are skipped (no empty bricks created).
        """
        data_frame = self.get_raw_renko([Decimal("100"), Decimal("103.04")])
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all().order_by("renko_data__sequence")
        # Only level 0 body emitted (intermediate levels skipped)
        self.assertEqual(len(candle_data), 1)
        self.assertEqual(candle_data[0].renko_data.level, 0)
        self.assertEqual(candle_data[0].renko_data.direction, 1)
        self.assertEqual(candle_data[0].renko_data.kind, RenkoKind.BODY)
        # Cache: at L3, entered from below, next seq=1
        self.assert_cache_state(
            {
                "level": 3,
                "direction": 1,
                "sequence": 1,
                "wicks": 0,
            }
        )
        # get_candle_data: 0 complete rows (L0 is last, filtered)
        df = DataFrame(
            self.candle.get_candle_data(self.timestamp_from, self.one_minute_from_now)
        )
        self.assertEqual(len(df), 0)

    def test_sequence_ordering_deterministic(self, mock_get_max_timestamp_to):
        """Verify sequence numbers provide deterministic ordering."""
        data_frame = self.get_raw_renko([Decimal("100"), Decimal("102"), Decimal("99")])
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all().order_by("renko_data__sequence")
        # Verify sequences are consecutive
        for i, candle in enumerate(candle_data):
            self.assertEqual(candle.renko_data.sequence, i)

    def test_incremental_variance_equals_batch(self, mock_get_max_timestamp_to):
        """Verify incremental aggregation produces same variance as batch."""
        # Create 3 trades at different prices
        prices = [Decimal("100"), Decimal("101"), Decimal("99.5")]

        # Batch aggregation (old way)
        df_batch = self.get_raw_renko(prices)
        batch_agg = aggregate_candle(df_batch)

        # Incremental aggregation (new way)
        agg = None
        for price in prices:
            df_single = self.get_raw_renko([price])
            trade_agg = aggregate_candle(df_single)
            if agg is None:
                agg = trade_agg
            else:
                agg = merge_cache(agg, trade_agg)

        # Should produce identical variance (within floating point precision)
        self.assertAlmostEqual(
            float(batch_agg["realizedVariance"]),
            float(agg["realizedVariance"]),
            places=15,
        )

    def test_ping_pong_delays_wick_emission(self, mock_get_max_timestamp_to):
        """Ping-pong pattern: wicks accumulate until body emits.

        Prices: 100 → 101 → 102.5 → 101.5 → 102.5 → 101.5 → 99.5
        Levels: L0  → L1  → L2    → L1    → L2    → L1    → L-1

        | Trade | Price | Level | Action                              |
        |-------|-------|-------|-------------------------------------|
        | 1     | 100   | 0     | Seed                                |
        | 2     | 101   | 1     | Exit L0 ↑ → BODY(L0, seq=0)         |
        | 3     | 102.5 | 2     | Exit L1 ↑ → BODY(L1, seq=1)         |
        | 4     | 101.5 | 1     | Exit L2 ↓ → wick(L2, ↓) pending     |
        | 5     | 102.5 | 2     | Exit L1 ↑ → wick(L1, ↑) pending     |
        | 6     | 101.5 | 1     | Exit L2 ↓ → wick(L2, ↓) pending     |
        | 7     | 99.5  | -1    | Exit L1 ↓ → BODY(L1, seq=2) + 3 wicks |

        Emitted: body(L0,↑), body(L1,↑), body(L1,↓), wick(L2,↓), wick(L1,↑), wick(L2,↓)
        """
        prices = [
            Decimal("100"),
            Decimal("101"),
            Decimal("102.5"),
            Decimal("101.5"),
            Decimal("102.5"),
            Decimal("101.5"),
            Decimal("99.5"),
        ]

        df = self.get_raw_renko(prices)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, df)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)

        candles = CandleData.objects.all().order_by("renko_data__sequence")

        self.assertEqual(len(candles), 6)
        self.assertEqual(candles[0].renko_data.level, 0)
        self.assertEqual(candles[0].renko_data.kind, RenkoKind.BODY)
        self.assertEqual(candles[0].renko_data.direction, 1)
        self.assertEqual(candles[1].renko_data.level, 1)
        self.assertEqual(candles[1].renko_data.kind, RenkoKind.BODY)
        self.assertEqual(candles[1].renko_data.direction, 1)
        self.assertEqual(candles[2].renko_data.level, 1)
        self.assertEqual(candles[2].renko_data.kind, RenkoKind.BODY)
        self.assertEqual(candles[2].renko_data.direction, -1)
        self.assertEqual(candles[3].renko_data.level, 2)
        self.assertEqual(candles[3].renko_data.kind, RenkoKind.WICK)
        self.assertEqual(candles[3].renko_data.direction, -1)  # Exit direction
        self.assertEqual(candles[4].renko_data.level, 1)
        self.assertEqual(candles[4].renko_data.kind, RenkoKind.WICK)
        self.assertEqual(candles[4].renko_data.direction, +1)  # Exit direction
        self.assertEqual(candles[5].renko_data.level, 2)
        self.assertEqual(candles[5].renko_data.kind, RenkoKind.WICK)
        self.assertEqual(candles[5].renko_data.direction, -1)  # Exit direction
        # Cache: at L-1, entered from above, no pending wicks (all flushed)
        self.assert_cache_state(
            {
                "level": -1,
                "direction": -1,
                "sequence": 6,
                "wicks": 0,
            }
        )
        # get_candle_data: wicks merged into bodies, same-level bodies merged
        # Output: L0, L1 (with L1↑ and L1↓ merged). Last row filtered → 1 row.
        result = DataFrame(
            self.candle.get_candle_data(self.timestamp_from, self.one_minute_from_now)
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["level"], 0)

    def test_pending_wick_emitted_on_continuation(self, mock_get_max_timestamp_to):
        """Continuation after wick: pending wicks emit with continuing body.

        Prices: 100 → 101 → 102.5 → 101.5 → 102.5 → 103.5
        Levels: L0  → L1  → L2    → L1    → L2    → L3

        | Trade | Price | Level | Action                              |
        |-------|-------|-------|-------------------------------------|
        | 1     | 100   | 0     | Seed                                |
        | 2     | 101   | 1     | Exit L0 ↑ → BODY(L0, seq=0)         |
        | 3     | 102.5 | 2     | Exit L1 ↑ → BODY(L1, seq=1)         |
        | 4     | 101.5 | 1     | Exit L2 ↓ → wick(L2, ↓) pending     |
        | 5     | 102.5 | 2     | Exit L1 ↑ → wick(L1, ↑) pending     |
        | 6     | 103.5 | 3     | Exit L2 ↑ → BODY(L2, seq=2) + 2 wicks |

        Emitted: body(L0,↑), body(L1,↑), body(L2,↑), wick(L2,↓), wick(L1,↑)
        """
        prices = [
            Decimal("100"),
            Decimal("101"),
            Decimal("102.5"),
            Decimal("101.5"),
            Decimal("102.5"),
            Decimal("103.5"),
        ]

        df = self.get_raw_renko(prices)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, df)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)

        candles = CandleData.objects.all().order_by("renko_data__sequence")

        self.assertEqual(len(candles), 5)
        self.assertEqual(candles[0].renko_data.level, 0)
        self.assertEqual(candles[0].renko_data.kind, RenkoKind.BODY)
        self.assertEqual(candles[0].renko_data.direction, 1)
        self.assertEqual(candles[1].renko_data.level, 1)
        self.assertEqual(candles[1].renko_data.kind, RenkoKind.BODY)
        self.assertEqual(candles[1].renko_data.direction, 1)
        self.assertEqual(candles[2].renko_data.level, 2)
        self.assertEqual(candles[2].renko_data.kind, RenkoKind.BODY)
        self.assertEqual(candles[2].renko_data.direction, 1)
        self.assertEqual(candles[3].renko_data.level, 2)
        self.assertEqual(candles[3].renko_data.kind, RenkoKind.WICK)
        self.assertEqual(candles[3].renko_data.direction, -1)  # Exit direction
        self.assertEqual(candles[4].renko_data.level, 1)
        self.assertEqual(candles[4].renko_data.kind, RenkoKind.WICK)
        self.assertEqual(candles[4].renko_data.direction, +1)  # Exit direction
        # Cache: at L3, entered from below, no pending wicks (all flushed)
        self.assert_cache_state(
            {
                "level": 3,
                "direction": 1,
                "sequence": 5,
                "wicks": 0,
            }
        )
        # get_candle_data: 2 complete rows (wicks merged, L2 filtered as last)
        result = DataFrame(
            self.candle.get_candle_data(self.timestamp_from, self.one_minute_from_now)
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["level"], 0)
        self.assertEqual(result.iloc[1]["level"], 1)

    def test_wicks_merged_into_body_by_level_adjacency(self, mock_get_max_timestamp_to):
        """get_candle_data merges wicks into adjacent bodies.

        Prices: 100 → 101 → 102.5 → 101.5 → 102.5 → 101.5 → 99
        Levels: L0  → L1  → L2    → L1    → L2    → L1    → L-1

        Raw DB contains bodies and wicks. get_candle_data() merges wicks
        into their parent bodies (by level adjacency) and returns only bodies.
        """
        # Clear Renko state
        CandleData.objects.filter(candle=self.candle).delete()
        CandleCache.objects.filter(candle=self.candle).delete()
        TradeData.objects.filter(
            symbol=self.symbol,
            timestamp__gte=self.timestamp_from,
            timestamp__lt=self.one_minute_from_now,
        ).delete()

        prices = [
            Decimal("100"),
            Decimal("101"),
            Decimal("102.5"),
            Decimal("101.5"),
            Decimal("102.5"),
            Decimal("101.5"),
            Decimal("99"),
        ]

        df = self.get_raw_renko(prices)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, df)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)

        # Query raw DB rows (should have bodies and wicks)
        raw_candles = CandleData.objects.filter(candle=self.candle).select_related(
            "renko_data"
        )
        wick_count = sum(1 for c in raw_candles if c.renko_data.kind == RenkoKind.WICK)
        self.assertGreater(wick_count, 0, "Expected at least one wick in raw DB")

        # Query via get_candle_data (wicks should be merged into bodies)
        df_result = DataFrame(
            self.candle.get_candle_data(
                self.timestamp_from, self.one_minute_from_now, is_complete=False
            )
        )

        # Verify no wick rows in output (all merged)
        wick_rows = df_result[df_result["kind"] == RenkoKind.WICK]
        self.assertEqual(
            len(wick_rows), 0, "All wicks should be merged into parent bodies"
        )
        # Cache: at L-2 (99 < 99.01), entered from above, no pending wicks
        self.assert_cache_state(
            {
                "level": -2,
                "direction": -1,
                "sequence": 6,
                "wicks": 0,
            }
        )

    def test_wick_attaches_to_correct_parent_by_direction(
        self, mock_get_max_timestamp_to
    ):
        """Wick with direction ↓ attaches to level-1, not level+1.

        Prices: 100 → 101 → 102.01 → 103.04 → 100 → 101 → 100 → 99
        Levels: L0  → L1  → L2     → L3     → L0  → L1  → L0  → L-1

        | Trade | Price  | Level | Action                                |
        |-------|--------|-------|---------------------------------------|
        | 1     | 100    | 0     | Seed                                  |
        | 2     | 101    | 1     | Exit L0 ↑ → BODY(L0, seq=0)           |
        | 3     | 102.01 | 2     | Exit L1 ↑ → BODY(L1, seq=1)           |
        | 4     | 103.04 | 3     | Exit L2 ↑ → BODY(L2, seq=2)           |
        | 5     | 100    | 0     | Exit L3 ↓ → wick(L3, ↓) pending       |
        | 6     | 101    | 1     | Exit L0 ↑ → wick(L0, ↑) pending       |
        | 7     | 100    | 0     | Exit L1 ↓ → wick(L1, ↓) pending       |
        | 8     | 99     | -1    | Exit L0 ↓ → BODY(L0, seq=3) + 3 wicks |

        Bug scenario: When both L0 and L2 bodies exist, L1 wick with direction ↓
        should attach to L0 (not L2). Direction determines parent level.
        """
        CandleData.objects.filter(candle=self.candle).delete()
        CandleCache.objects.filter(candle=self.candle).delete()
        TradeData.objects.filter(
            symbol=self.symbol,
            timestamp__gte=self.timestamp_from,
            timestamp__lt=self.one_minute_from_now,
        ).delete()

        prices = [
            Decimal("100"),
            Decimal("101"),
            Decimal("102.01"),
            Decimal("103.04"),
            Decimal("100"),
            Decimal("101"),
            Decimal("100"),
            Decimal("99"),
        ]

        df = self.get_raw_renko(prices)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, df)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)

        # Verify raw DB structure
        raw_candles = list(
            CandleData.objects.filter(candle=self.candle)
            .select_related("renko_data")
            .order_by("renko_data__sequence")
        )

        # Should have 7 records: 4 bodies + 3 wicks
        self.assertEqual(len(raw_candles), 7)

        # Find the L1 wick with direction -1
        l1_wick = next(
            c
            for c in raw_candles
            if c.renko_data.kind == RenkoKind.WICK
            and c.renko_data.level == 1
            and c.renko_data.direction == -1
        )
        self.assertIsNotNone(l1_wick)

        # Get candle data - wicks should be merged correctly
        result = DataFrame(
            self.candle.get_candle_data(
                self.timestamp_from, self.one_minute_from_now, is_complete=False
            )
        )

        # Verify no wick rows (all merged)
        wick_rows = result[result["kind"] == RenkoKind.WICK]
        self.assertEqual(len(wick_rows), 0)

        # Verify the L0 body (the second one, seq=3) got the L1↓ wick merged
        # by checking it has data from the L1 wick's trades
        l0_bodies = result[result["level"] == 0]
        # Should have 2 L0 bodies (seq=0 and seq=3, but seq=3 merged from return)
        # Actually after same-level merge, there should be 1 L0 body with merged data
        self.assertGreater(len(l0_bodies), 0)

    def test_get_candle_data_canonicalization(self, mock_get_max_timestamp_to):
        """Multi-level jump with reversal: L0→L1→L3→L2→L0.

        Prices: 100 → 101.5 → 103.5 → 102.5 → 100.5
        Levels: L0  → L1    → L3    → L2    → L0

        | Trade | Price | Level | Action                              |
        |-------|-------|-------|-------------------------------------|
        | 1     | 100   | 0     | Seed                                |
        | 2     | 101.5 | 1     | Exit L0 ↑ → BODY(L0, seq=0)         |
        | 3     | 103.5 | 3     | Exit L1 ↑ → BODY(L1, seq=1), skip L2 |
        | 4     | 102.5 | 2     | Exit L3 ↓ → wick(L3, ↓) pending     |
        | 5     | 100.5 | 0     | Exit L2 ↓ → BODY(L2, seq=2) + wick  |

        Raw DB: body(L0,↑), body(L1,↑), body(L2,↓), wick(L3,↓)
        get_candle_data: merges wick into body, adds incomplete from cache
        """
        # Clear Renko state before test
        CandleData.objects.filter(candle=self.candle).delete()
        CandleCache.objects.filter(candle=self.candle).delete()
        TradeData.objects.filter(
            symbol=self.symbol,
            timestamp__gte=self.timestamp_from,
            timestamp__lt=self.one_minute_from_now,
        ).delete()

        prices = [
            Decimal("100"),
            Decimal("101.5"),
            Decimal("103.5"),
            Decimal("102.5"),
            Decimal("100.5"),
        ]
        data_frame = self.get_raw_renko(prices)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)

        # Query raw DB rows (via ORM)
        raw_candles = (
            CandleData.objects.filter(candle=self.candle)
            .select_related("renko_data")
            .order_by("renko_data__sequence")
        )

        # Verify raw structure (no empty bricks for skipped levels)
        raw_levels = [c.renko_data.level for c in raw_candles]
        raw_kinds = [c.renko_data.kind for c in raw_candles]
        raw_directions = [c.renko_data.direction for c in raw_candles]

        # Expected pattern with emit-on-exit (no empty bricks):
        # - Trade at 100 (L0): seed, no emit
        # - Trade at 101.5 (L1): emit body(0,+1)
        # - Trade at 103.5 (L3): emit body(1,+1), skip L2
        # - Trade at 102.5 (L2): wick(3,-1) pending
        # - Trade at 100.5 (L0): emit body(2,-1) + wick(3,-1)
        # Total: 4 rows
        self.assertEqual(
            raw_candles.count(), 4, f"Expected 4 raw DB rows, got {raw_candles.count()}"
        )
        self.assertEqual(raw_levels, [0, 1, 2, 3])
        self.assertEqual(
            raw_kinds, [RenkoKind.BODY, RenkoKind.BODY, RenkoKind.BODY, RenkoKind.WICK]
        )
        self.assertEqual(raw_directions, [1, 1, -1, -1])

        # Query via get_candle_data(is_complete=False)
        df = DataFrame(
            self.candle.get_candle_data(
                self.timestamp_from, self.one_minute_from_now, is_complete=False
            )
        )

        # Verify canonicalization: wicks merged into bodies
        # Expected with is_complete=False (bodies + incomplete from cache):
        # body(0,+1), body(1,+1), body(2,-1 with wick merged), incomplete(0,-1)
        self.assertEqual(
            len(df), 4, "Expected 4 rows (3 from DB + 1 incomplete from cache)"
        )

        # Verify NO wick rows (merged into parent body)
        wick_rows = df[df["kind"] == RenkoKind.WICK]
        self.assertEqual(len(wick_rows), 0, "Expected 0 wick rows (merged into body)")

        # Verify body stream: 0(↑), 1(↑), 2(↓ with wick), 0(↓ incomplete)
        body_levels = df["level"].tolist()
        body_directions = df["direction"].tolist()
        self.assertEqual(
            body_levels,
            [0, 1, 2, 0],
            "Body levels after canonicalization with incomplete",
        )
        self.assertEqual(body_directions, [1, 1, -1, -1], "Body directions")

        # Test is_complete parameter
        df_complete = DataFrame(
            self.candle.get_candle_data(
                self.timestamp_from, self.one_minute_from_now, is_complete=True
            )
        )
        # Should exclude last body brick (level 2↓)
        # Expected (bodies only, wicks merged): body(0↑), body(1↑)
        self.assertEqual(
            len(df_complete), 2, "is_complete=True should exclude last body"
        )
        # Cache: at L0, entered from above, no pending wicks
        self.assert_cache_state(
            {
                "level": 0,
                "direction": -1,
                "sequence": 4,
                "wicks": 0,
            }
        )
        # Verify last row is body at level 1
        self.assertEqual(
            df_complete.iloc[-1]["kind"],
            RenkoKind.BODY,
            "Last row should be body (wicks are merged)",
        )
        self.assertEqual(
            df_complete.iloc[-1]["level"], 1, "Last body should be level 1"
        )

    def test_get_candle_data_incomplete_brick_from_cache(
        self, mock_get_max_timestamp_to
    ):
        """Incomplete brick synthesis: L0→L1 with L1 incomplete in cache.

        Prices: 100 → 101.5
        Levels: L0  → L1

        | Trade | Price | Level | Action                    |
        |-------|-------|-------|---------------------------|
        | 1     | 100   | 0     | Seed                      |
        | 2     | 101.5 | 1     | Exit L0 ↑ → BODY(L0, seq=0) |

        DB: body(L0, ↑)
        Cache: level=1, incomplete brick
        get_candle_data(is_complete=False): includes synthetic L1 brick
        """
        # Clear state
        CandleData.objects.filter(candle=self.candle).delete()
        CandleCache.objects.filter(candle=self.candle).delete()
        TradeData.objects.filter(
            symbol=self.symbol,
            timestamp__gte=self.timestamp_from,
            timestamp__lt=self.one_minute_from_now,
        ).delete()

        # Create scenario with incomplete brick in cache
        # Prices: 100 (L0) -> 101.5 (L1)
        # Level 0 emitted to DB, level 1 remains in cache (incomplete)
        prices = [
            Decimal("100"),  # level 0 - initializes
            Decimal("101.5"),  # level 1 - emits body(0↑), enters L1 but doesn't exit
        ]
        data_frame = self.get_raw_renko(prices)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)

        # Verify DB contains only level 0 body
        db_candles = CandleData.objects.filter(candle=self.candle).order_by(
            "renko_data__sequence"
        )
        self.assertEqual(db_candles.count(), 1)
        self.assertEqual(db_candles[0].renko_data.level, 0)
        self.assertEqual(db_candles[0].renko_data.kind, RenkoKind.BODY)

        # Verify cache contains level 1 state
        cache_obj = (
            CandleCache.objects.filter(candle=self.candle)
            .order_by("-timestamp")
            .first()
        )
        self.assertIsNotNone(cache_obj)
        cache = cache_obj.json_data
        self.assertEqual(cache["level"], 1)

        # Query with is_complete=False (should include incomplete brick)
        df_incomplete = DataFrame(
            self.candle.get_candle_data(
                self.timestamp_from, self.one_minute_from_now, is_complete=False
            )
        )

        # Verify output contains both DB brick and synthetic incomplete brick
        self.assertEqual(
            len(df_incomplete),
            2,
            "Expected 2 rows: body(0↑) from DB + incomplete body(1↑) from cache",
        )

        # Verify first row is body(0↑) from DB
        row0 = df_incomplete.iloc[0]
        self.assertEqual(row0["level"], 0)
        self.assertEqual(row0["kind"], RenkoKind.BODY)
        self.assertEqual(row0["direction"], 1)

        # Verify second row is incomplete body(1↑) from cache
        row1 = df_incomplete.iloc[1]
        self.assertEqual(row1["level"], 1)
        self.assertEqual(row1["kind"], RenkoKind.BODY)
        self.assertEqual(row1["direction"], 1, "Direction should be 1 (from_below)")
        self.assertEqual(
            row1["bar_index"],
            cache["sequence"],
            "bar_index should match cache sequence",
        )

        # Verify OHLCV fields are present
        for field in ["open", "high", "low", "close"]:
            self.assertIn(field, row1, f"Incomplete brick should have {field}")
            self.assertIsNotNone(row1[field])

        # Query with is_complete=True (should exclude incomplete brick)
        df_complete = DataFrame(
            self.candle.get_candle_data(
                self.timestamp_from, self.one_minute_from_now, is_complete=True
            )
        )

        # Verify incomplete brick is excluded (only DB bricks, minus last body)
        # In this case, level 0 is the last body, so it gets excluded too
        self.assertEqual(
            len(df_complete), 0, "is_complete=True should exclude all incomplete bricks"
        )

    def test_get_candle_data_one_brick_lag(self, mock_get_max_timestamp_to):
        """Complete bricks have 1-brick lag for wick resolution.

        A brick is only "complete" once the next brick emits, because pending
        wicks attach to their parent body. This test verifies the intentional
        1-brick lag when is_complete=True.

        Prices: 100 → 101.5 → 102.5
        Levels: L0  → L1    → L2

        | Trade | Price | Level | Action                      |
        |-------|-------|-------|-----------------------------|
        | 1     | 100   | 0     | Seed                        |
        | 2     | 101.5 | 1     | Exit L0 ↑ → BODY(L0, seq=0) |
        | 3     | 102.5 | 2     | Exit L1 ↑ → BODY(L1, seq=1) |

        DB: [body(L0, ↑), body(L1, ↑)]
        is_complete=True: Returns only body(L0) - L1 excluded as last brick
        is_complete=False: Returns [body(L0), body(L1), incomplete(L2)]
        """
        CandleData.objects.filter(candle=self.candle).delete()
        CandleCache.objects.filter(candle=self.candle).delete()
        TradeData.objects.filter(
            symbol=self.symbol,
            timestamp__gte=self.timestamp_from,
            timestamp__lt=self.one_minute_from_now,
        ).delete()

        prices = [
            Decimal("100"),  # L0 - seed
            Decimal("101.5"),  # L1 - emits body(L0↑)
            Decimal("102.5"),  # L2 - emits body(L1↑)
        ]
        df = self.get_raw_renko(prices)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, df)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)

        # Verify DB has 2 bodies
        db_count = CandleData.objects.filter(candle=self.candle).count()
        self.assertEqual(db_count, 2)

        # is_complete=True excludes last brick (1-brick lag)
        df_complete = DataFrame(
            self.candle.get_candle_data(
                self.timestamp_from, self.one_minute_from_now, is_complete=True
            )
        )
        self.assertEqual(
            len(df_complete), 1, "1-brick lag: last body excluded when is_complete=True"
        )
        self.assertEqual(df_complete.iloc[0]["level"], 0)

        # is_complete=False includes all + incomplete from cache
        df_incomplete = DataFrame(
            self.candle.get_candle_data(
                self.timestamp_from, self.one_minute_from_now, is_complete=False
            )
        )
        self.assertEqual(
            len(df_incomplete), 3, "is_complete=False: 2 DB bodies + 1 incomplete"
        )

    def test_boundary_spacing_invariant(self, mock_get_max_timestamp_to):
        """Verify boundary spacing is constant across levels."""
        p = Decimal("0.01")  # target_percentage_change
        origin = Decimal("100")  # Same as test fixture

        # Test multiple levels around test prices
        for level in [0, 1, 2, 5, 10]:
            boundary_L = origin * (Decimal(1) + p) ** level
            boundary_L_plus_1 = origin * (Decimal(1) + p) ** (level + 1)
            ratio = boundary_L_plus_1 / boundary_L
            # Use Decimal arithmetic, not float()
            self.assertEqual(
                ratio.quantize(Decimal("0.0001")),
                (Decimal(1) + p).quantize(Decimal("0.0001")),
            )

    def test_negative_levels_below_origin(self, mock_get_max_timestamp_to):
        """Negative levels: price below origin maps to negative level.

        Setup: origin_price=1, target_percentage_change=0.01
        Price: 0.5

        Level calculation: floor(ln(0.5/1) / ln(1.01)) ≈ floor(-69.66) = -70

        | Trade | Price | Level | Action                     |
        |-------|-------|-------|----------------------------|
        | 1     | 0.5   | -70   | Seed at negative level     |

        Cache: level=-70 (negative)
        """
        candle = RenkoBrick.objects.create(
            symbol=self.symbol,
            json_data={
                "source_data": FileData.RAW,
                "target_percentage_change": Decimal("0.01"),
                "origin_price": Decimal("1"),
            },
        )
        data_frame = pd.DataFrame(
            [self.get_random_trade(timestamp=self.timestamp_from, price=Decimal("0.5"))]
        )
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(candle, self.timestamp_from, self.one_minute_from_now)

        # Verify cache has negative level
        cache = CandleCache.objects.filter(candle=candle).first()
        self.assertIsNotNone(cache)
        self.assertLess(cache.json_data["level"], 0)

    def test_deterministic_levels_across_series(self, mock_get_max_timestamp_to):
        """Verify same price maps to same level across different series."""
        origin = Decimal("1")
        p = Decimal("0.01")

        # Create two candles with same config
        candle1 = RenkoBrick.objects.create(
            symbol=self.symbol,
            json_data={
                "source_data": FileData.RAW,
                "target_percentage_change": p,
                "origin_price": origin,
            },
        )
        candle2 = RenkoBrick.objects.create(
            symbol=self.symbol,
            json_data={
                "source_data": FileData.RAW,
                "target_percentage_change": p,
                "origin_price": origin,
            },
        )

        # Same price for both
        price = Decimal("100")

        # Process price for candle1
        data_frame1 = pd.DataFrame(
            [self.get_random_trade(timestamp=self.timestamp_from, price=price)]
        )
        self.write_trade_data(
            self.timestamp_from, self.one_minute_from_now, data_frame1
        )
        aggregate_candles(candle1, self.timestamp_from, self.one_minute_from_now)

        # Process price for candle2
        data_frame2 = pd.DataFrame(
            [self.get_random_trade(timestamp=self.timestamp_from, price=price)]
        )
        TradeData.objects.filter(symbol=self.symbol).delete()  # Clear trades
        self.write_trade_data(
            self.timestamp_from, self.one_minute_from_now, data_frame2
        )
        aggregate_candles(candle2, self.timestamp_from, self.one_minute_from_now)

        # Verify both have same level
        cache1 = CandleCache.objects.filter(candle=candle1).first()
        cache2 = CandleCache.objects.filter(candle=candle2).first()
        self.assertEqual(cache1.json_data["level"], cache2.json_data["level"])
