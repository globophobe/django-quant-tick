from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import time_machine
from django.test import TestCase
from pandas import DataFrame

from quant_tick.constants import Exchange, FileData, Frequency, RenkoKind, SampleType
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
    Symbol,
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
        self.candle = self.get_candle()
        self.symbol = self.get_symbol()
        self.candle.symbols.add(self.symbol)

    def get_candle(self) -> Candle:
        """Get candle."""
        return Candle.objects.create(json_data={"source_data": FileData.RAW})

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
class CandleTest(BaseDayIteratorTest, TestCase):
    """Candle test."""

    def setUp(self):
        """Set up."""
        super().setUp()
        self.candle = Candle.objects.create(json_data={"source_data": FileData.RAW})

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
            json_data={"source_data": FileData.RAW, "window": "1min"}
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
            json_data={"source_data": FileData.RAW, "window": "2min"}
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
            json_data={"source_data": FileData.RAW, "window": "1h"}
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
            json_data={"source_data": FileData.RAW, "window": "2h"}
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
            json_data={
                "source_data": FileData.RAW,
                "sample_type": SampleType.NOTIONAL,
                "target_value": 1,
            }
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
            json_data={
                "source_data": FileData.RAW,
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
            json_data={
                "source_data": FileData.RAW,
                "sample_type": SampleType.NOTIONAL,
                "moving_average_number_of_days": 1,
                "target_candles_per_day": 1,
            }
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
    return_value=datetime(2009, 1, 4, 0, 3).replace(tzinfo=timezone.utc),
)
class TimeBasedMultiExchangeCandleTest(
    BaseMinuteIteratorTest,
    BaseWriteTradeDataTest,
    BaseCandleCacheIteratorTest,
    TestCase,
):
    """Time based candle with multiple exchanges."""

    def get_candle(self) -> Candle:
        """Get candle."""
        return TimeBasedCandle.objects.create(
            json_data={"source_data": FileData.RAW, "window": "1min"}
        )

    def get_second_symbol(self) -> Symbol:
        """Get second symbol on different exchange."""
        return Symbol.objects.create(
            global_symbol=self.global_symbol,
            exchange=Exchange.BINANCE,
            api_symbol="test-binance",
            save_raw=True,
        )

    def test_multi_exchange_candle(self, mock_get_max_timestamp_to):
        """Multi exchange time based candle."""
        symbol_2 = self.get_second_symbol()
        self.candle.symbols.add(symbol_2)

        filtered_1 = self.get_filtered(self.timestamp_from, price=100, notional=1)
        TradeData.write(
            self.symbol,
            self.timestamp_from,
            self.one_minute_from_now,
            filtered_1,
            pd.DataFrame([]),
        )

        filtered_2 = self.get_filtered(self.timestamp_from, price=101, notional=2)
        TradeData.write(
            symbol_2,
            self.timestamp_from,
            self.one_minute_from_now,
            filtered_2,
            pd.DataFrame([]),
        )

        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)

        data = candle_data[0].json_data
        # Nested structure with exchanges dict
        self.assertIn("exchanges", data)
        self.assertIn("coinbase", data["exchanges"])
        self.assertIn("binance", data["exchanges"])
        # Per-exchange fields
        self.assertIn("open", data["exchanges"]["coinbase"])
        self.assertIn("volume", data["exchanges"]["coinbase"])
        self.assertIn("realizedVariance", data["exchanges"]["coinbase"])
        self.assertIn("open", data["exchanges"]["binance"])
        self.assertIn("volume", data["exchanges"]["binance"])
        self.assertIn("realizedVariance", data["exchanges"]["binance"])

    def test_multi_exchange_cache_merge(self, mock_get_max_timestamp_to):
        """Multi exchange time based candle cache merge."""
        # Use 2-minute window to trigger cache merge
        self.candle.json_data["window"] = "2min"
        self.candle.save()

        symbol_2 = self.get_second_symbol()
        self.candle.symbols.add(symbol_2)

        # Minute 1: coinbase=100, binance=200
        filtered_cb_1 = self.get_filtered(self.timestamp_from, price=100, notional=1)
        TradeData.write(
            self.symbol,
            self.timestamp_from,
            self.one_minute_from_now,
            filtered_cb_1,
            pd.DataFrame([]),
        )
        filtered_bn_1 = self.get_filtered(self.timestamp_from, price=200, notional=2)
        TradeData.write(
            symbol_2,
            self.timestamp_from,
            self.one_minute_from_now,
            filtered_bn_1,
            pd.DataFrame([]),
        )

        # Minute 2: coinbase=110, binance=190
        filtered_cb_2 = self.get_filtered(
            self.one_minute_from_now, price=110, notional=1
        )
        TradeData.write(
            self.symbol,
            self.one_minute_from_now,
            self.two_minutes_from_now,
            filtered_cb_2,
            pd.DataFrame([]),
        )
        filtered_bn_2 = self.get_filtered(
            self.one_minute_from_now, price=190, notional=2
        )
        TradeData.write(
            symbol_2,
            self.one_minute_from_now,
            self.two_minutes_from_now,
            filtered_bn_2,
            pd.DataFrame([]),
        )

        # Aggregate both minutes
        for i in range(2):
            aggregate_candles(
                self.candle,
                self.timestamp_from + pd.Timedelta(f"{i}min"),
                self.one_minute_from_now + pd.Timedelta(f"{i}min"),
            )

        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)

        data = candle_data[0].json_data
        cb = data["exchanges"]["coinbase"]
        bn = data["exchanges"]["binance"]
        # Per-exchange open from first segment.
        self.assertEqual(cb["open"], 100)
        self.assertEqual(bn["open"], 200)
        # Per-exchange close from second segment.
        self.assertEqual(cb["close"], 110)
        self.assertEqual(bn["close"], 190)
        # Per-exchange volume summed.
        cb_vol_1 = filtered_cb_1["volume"].sum()
        cb_vol_2 = filtered_cb_2["volume"].sum()
        bn_vol_1 = filtered_bn_1["volume"].sum()
        bn_vol_2 = filtered_bn_2["volume"].sum()
        self.assertEqual(cb["volume"], cb_vol_1 + cb_vol_2)
        self.assertEqual(bn["volume"], bn_vol_1 + bn_vol_2)
        # Per-exchange realized variance exists.
        self.assertIn("realizedVariance", cb)
        self.assertIn("realizedVariance", bn)


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=timezone.utc),
)
class ConstantMultiExchangeCandleTest(
    BaseHourIteratorTest, BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):
    """Constant candle with multiple exchanges."""

    def get_candle(self) -> Candle:
        """Get candle."""
        return ConstantCandle.objects.create(
            json_data={
                "source_data": FileData.RAW,
                "sample_type": SampleType.NOTIONAL,
                "target_value": 2,
            }
        )

    def get_second_symbol(self) -> Symbol:
        """Get second symbol on different exchange."""
        return Symbol.objects.create(
            global_symbol=self.global_symbol,
            exchange=Exchange.BINANCE,
            api_symbol="test-binance",
            save_raw=True,
        )

    def test_multi_exchange_candle(self, mock_get_max_timestamp_to):
        """Multi exchange constant candle."""
        symbol_2 = self.get_second_symbol()
        self.candle.symbols.add(symbol_2)

        filtered_1 = self.get_filtered(self.timestamp_from, price=100, notional=1)
        TradeData.write(
            self.symbol,
            self.timestamp_from,
            self.one_hour_from_now,
            filtered_1,
            pd.DataFrame([]),
        )

        filtered_2 = self.get_filtered(self.timestamp_from, price=101, notional=1)
        TradeData.write(
            symbol_2,
            self.timestamp_from,
            self.one_hour_from_now,
            filtered_2,
            pd.DataFrame([]),
        )

        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)

        data = candle_data[0].json_data
        # Nested structure with exchanges dict
        self.assertIn("exchanges", data)
        self.assertIn("coinbase", data["exchanges"])
        self.assertIn("binance", data["exchanges"])
        # Per-exchange fields
        self.assertIn("open", data["exchanges"]["coinbase"])
        self.assertIn("volume", data["exchanges"]["coinbase"])
        self.assertIn("realizedVariance", data["exchanges"]["coinbase"])
        self.assertIn("open", data["exchanges"]["binance"])
        self.assertIn("volume", data["exchanges"]["binance"])
        self.assertIn("realizedVariance", data["exchanges"]["binance"])

    def test_multi_exchange_cache_merge(self, mock_get_max_timestamp_to):
        """Multi exchange constant candle cache merge."""
        # Higher target so data accumulates in cache before candle completes
        self.candle.json_data["target_value"] = 10
        self.candle.save()

        symbol_2 = self.get_second_symbol()
        self.candle.symbols.add(symbol_2)

        # First batch: coinbase=100, binance=200 (notional=3 total, below target)
        filtered_cb_1 = self.get_filtered(
            self.timestamp_from, price=100, notional=Decimal("1")
        )
        TradeData.write(
            self.symbol,
            self.timestamp_from,
            self.one_hour_from_now,
            filtered_cb_1,
            pd.DataFrame([]),
        )
        filtered_bn_1 = self.get_filtered(
            self.timestamp_from, price=200, notional=Decimal("2")
        )
        TradeData.write(
            symbol_2,
            self.timestamp_from,
            self.one_hour_from_now,
            filtered_bn_1,
            pd.DataFrame([]),
        )

        # Second batch: coinbase=110, binance=190 (notional=8 more, total=11 exceeds target)
        filtered_cb_2 = self.get_filtered(
            self.one_hour_from_now, price=110, notional=Decimal("3")
        )
        TradeData.write(
            self.symbol,
            self.one_hour_from_now,
            self.two_hours_from_now,
            filtered_cb_2,
            pd.DataFrame([]),
        )
        filtered_bn_2 = self.get_filtered(
            self.one_hour_from_now, price=190, notional=Decimal("5")
        )
        TradeData.write(
            symbol_2,
            self.one_hour_from_now,
            self.two_hours_from_now,
            filtered_bn_2,
            pd.DataFrame([]),
        )

        # Aggregate both hours
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        aggregate_candles(self.candle, self.one_hour_from_now, self.two_hours_from_now)

        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)

        data = candle_data[0].json_data
        cb = data["exchanges"]["coinbase"]
        bn = data["exchanges"]["binance"]
        self.assertEqual(cb["open"], 100)
        self.assertEqual(bn["open"], 200)
        self.assertIn("volume", cb)
        self.assertIn("volume", bn)
        self.assertIn("realizedVariance", cb)
        self.assertIn("realizedVariance", bn)


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.controllers.iterators.CandleCacheIterator.get_max_timestamp_to",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=timezone.utc),
)
class AdaptiveMultiExchangeCandleTest(
    BaseHourIteratorTest, BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):
    """Adaptive candle with multiple exchanges."""

    def setUp(self):
        """Set up."""
        super().setUp()
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=get_min_time(self.timestamp_from, value="1d")
            - pd.Timedelta("1d"),
            frequency=Frequency.DAY,
            json_data={"candle": {"notional": 10}},
        )

    def get_candle(self) -> Candle:
        """Get candle."""
        return AdaptiveCandle.objects.create(
            json_data={
                "source_data": FileData.RAW,
                "sample_type": SampleType.NOTIONAL,
                "moving_average_number_of_days": 1,
                "target_candles_per_day": 1,
            }
        )

    def get_second_symbol(self) -> Symbol:
        """Get second symbol on different exchange."""
        return Symbol.objects.create(
            global_symbol=self.global_symbol,
            exchange=Exchange.BINANCE,
            api_symbol="test-binance",
            save_raw=True,
        )

    def test_multi_exchange_candle(self, mock_get_max_timestamp_to):
        """Multi exchange adaptive candle."""
        symbol_2 = self.get_second_symbol()
        self.candle.symbols.add(symbol_2)

        # Add historical data for symbol_2 so can_aggregate passes
        # Total MA = 10+10 = 20, target = 20 / 1 day / 1 candle = 20
        TradeData.objects.create(
            symbol=symbol_2,
            timestamp=get_min_time(self.timestamp_from, value="1d")
            - pd.Timedelta("1d"),
            frequency=Frequency.DAY,
            json_data={"candle": {"notional": 10}},
        )

        # With target_value=20, use notional=10+11=21 to exceed threshold
        filtered_1 = self.get_filtered(
            self.timestamp_from, price=100, notional=Decimal("10")
        )
        TradeData.write(
            self.symbol,
            self.timestamp_from,
            self.one_hour_from_now,
            filtered_1,
            pd.DataFrame([]),
        )

        filtered_2 = self.get_filtered(
            self.timestamp_from, price=101, notional=Decimal("11")
        )
        TradeData.write(
            symbol_2,
            self.timestamp_from,
            self.one_hour_from_now,
            filtered_2,
            pd.DataFrame([]),
        )

        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)

        data = candle_data[0].json_data
        # Nested structure with exchanges dict
        self.assertIn("exchanges", data)
        self.assertIn("coinbase", data["exchanges"])
        self.assertIn("binance", data["exchanges"])
        # Per-exchange fields
        self.assertIn("open", data["exchanges"]["coinbase"])
        self.assertIn("volume", data["exchanges"]["coinbase"])
        self.assertIn("realizedVariance", data["exchanges"]["coinbase"])
        self.assertIn("open", data["exchanges"]["binance"])
        self.assertIn("volume", data["exchanges"]["binance"])
        self.assertIn("realizedVariance", data["exchanges"]["binance"])

    def test_multi_exchange_cache_merge(self, mock_get_max_timestamp_to):
        """Multi exchange adaptive candle cache merge."""
        symbol_2 = self.get_second_symbol()
        self.candle.symbols.add(symbol_2)

        # Add historical data for symbol_2 so can_aggregate passes
        # Total MA = 10+10 = 20, target = 20 / 1 day / 1 candle = 20
        TradeData.objects.create(
            symbol=symbol_2,
            timestamp=get_min_time(self.timestamp_from, value="1d")
            - pd.Timedelta("1d"),
            frequency=Frequency.DAY,
            json_data={"candle": {"notional": 10}},
        )

        # First batch: coinbase=100, binance=200 (notional=12 total, below target of 20)
        filtered_cb_1 = self.get_filtered(
            self.timestamp_from, price=100, notional=Decimal("5")
        )
        TradeData.write(
            self.symbol,
            self.timestamp_from,
            self.one_hour_from_now,
            filtered_cb_1,
            pd.DataFrame([]),
        )
        filtered_bn_1 = self.get_filtered(
            self.timestamp_from, price=200, notional=Decimal("7")
        )
        TradeData.write(
            symbol_2,
            self.timestamp_from,
            self.one_hour_from_now,
            filtered_bn_1,
            pd.DataFrame([]),
        )

        # Second batch: coinbase=110, binance=190 (notional=12 more, total=24 exceeds target)
        filtered_cb_2 = self.get_filtered(
            self.one_hour_from_now, price=110, notional=Decimal("6")
        )
        TradeData.write(
            self.symbol,
            self.one_hour_from_now,
            self.two_hours_from_now,
            filtered_cb_2,
            pd.DataFrame([]),
        )
        filtered_bn_2 = self.get_filtered(
            self.one_hour_from_now, price=190, notional=Decimal("6")
        )
        TradeData.write(
            symbol_2,
            self.one_hour_from_now,
            self.two_hours_from_now,
            filtered_bn_2,
            pd.DataFrame([]),
        )

        # Aggregate both hours.
        aggregate_candles(self.candle, self.timestamp_from, self.one_hour_from_now)
        aggregate_candles(self.candle, self.one_hour_from_now, self.two_hours_from_now)

        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)

        data = candle_data[0].json_data
        cb = data["exchanges"]["coinbase"]
        bn = data["exchanges"]["binance"]
        # Per-exchange open from first segment, close from last.
        self.assertEqual(cb["open"], 100)
        self.assertEqual(bn["open"], 200)
        self.assertIn("volume", cb)
        self.assertIn("volume", bn)
        self.assertIn("realizedVariance", cb)
        self.assertIn("realizedVariance", bn)


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
            json_data={
                "source_data": FileData.RAW,
                "target_percentage_change": Decimal("0.01"),
                "origin_price": Decimal("100"),  # Keep existing level expectations (prices ~100)
            }
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

    def test_one_trade_one_brick(self, mock_get_max_timestamp_to):
        """One trade, no brick (emit-on-exit: first trade initializes but doesn't emit)."""
        data_frame = self.get_raw_renko(Decimal("100"))
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 0)

    def test_two_trades_one_brick_up(self, mock_get_max_timestamp_to):
        """Two trades, one brick (emit-on-exit: level 0 emitted when moving to level 1)."""
        data_frame = self.get_raw_renko([Decimal("100"), Decimal("101")])
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        brick = candle_data[0].json_data
        brick_renko = candle_data[0].renko_data
        brick_renko = candle_data[0].renko_data
        self.assertEqual(brick_renko.level, 0)
        self.assertEqual(brick_renko.direction, 1)
        self.assertEqual(brick_renko.kind, RenkoKind.BODY)
        self.assertEqual(brick["close"], data_frame.iloc[0].price)

    def test_three_trades_one_brick_up(self, mock_get_max_timestamp_to):
        """Three trades, one brick (third trade stays in level 1)."""
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

    def test_three_trades_one_brick_up_incomplete_reversal(
        self, mock_get_max_timestamp_to
    ):
        """Three trades: up to level 1, then back to level 0 (pending wick not emitted)."""
        data_frame = self.get_raw_renko(
            [Decimal("100"), Decimal("101"), Decimal("100")]
        )
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all()
        # Only 1 brick emitted: level 0 body
        # Level 1 wick remains pending (not emitted because no parent body emitted)
        self.assertEqual(candle_data.count(), 1)
        brick0 = candle_data[0].json_data
        brick0_renko = candle_data[0].renko_data
        # Brick 0: level 0 exited upward (body)
        self.assertEqual(brick0_renko.level, 0)
        self.assertEqual(brick0_renko.direction, 1)
        self.assertEqual(brick0_renko.kind, RenkoKind.BODY)

    def test_three_trades_one_brick_up_reversal_down(self, mock_get_max_timestamp_to):
        """Three trades: multi-level jump from level 1 to -1 (emits 3 bricks)."""
        data_frame = self.get_raw_renko([Decimal("100"), Decimal("101"), Decimal("99.5")])
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 3)
        brick0 = candle_data[0].json_data
        brick0_renko = candle_data[0].renko_data
        brick1 = candle_data[1].json_data
        brick1_renko = candle_data[1].renko_data
        brick2 = candle_data[2].json_data
        brick2_renko = candle_data[2].renko_data
        # Brick 0: level 0 exited upward (body)
        self.assertEqual(brick0_renko.level, 0)
        self.assertEqual(brick0_renko.direction, 1)
        # Brick 1: level 1 wick (upper wick relative to parent level 0)
        self.assertEqual(brick1_renko.level, 1)
        self.assertEqual(brick1_renko.direction, +1)  # Upper wick, side-based
        self.assertEqual(brick1_renko.kind, RenkoKind.WICK)
        # Brick 2: level 0 jumped over (empty body brick)
        self.assertEqual(brick2_renko.level, 0)
        self.assertEqual(brick2_renko.direction, -1)
        self.assertEqual(brick2_renko.kind, RenkoKind.BODY)

    def test_logarithmic_reversibility(self, mock_get_max_timestamp_to):
        """Verify logarithmic levels with multi-level jump (level 1 to -1 emits 3 bricks)."""
        data_frame = self.get_raw_renko([Decimal("100"), Decimal("101"), Decimal("99.5")])
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all().order_by("renko_data__sequence")
        self.assertEqual(len(candle_data), 3)
        brick0 = candle_data[0].json_data
        brick0_renko = candle_data[0].renko_data
        brick1 = candle_data[1].json_data
        brick1_renko = candle_data[1].renko_data
        brick2 = candle_data[2].json_data
        brick2_renko = candle_data[2].renko_data
        # Brick 0: seed level 0 exited upward (body)
        self.assertEqual(brick0_renko.level, 0)
        self.assertEqual(brick0_renko.direction, 1)
        self.assertEqual(brick0_renko.kind, RenkoKind.BODY)
        # Brick 1: level 1 wick (upper wick relative to parent level 0)
        self.assertEqual(brick1_renko.level, 1)
        self.assertEqual(brick1_renko.direction, +1)  # Upper wick, side-based
        self.assertEqual(brick1_renko.kind, RenkoKind.WICK)
        # Brick 2: level 0 jumped over (empty body)
        self.assertEqual(brick2_renko.level, 0)
        self.assertEqual(brick2_renko.direction, -1)
        self.assertEqual(brick2_renko.kind, RenkoKind.BODY)

    def test_small_move_no_phantom_brick(self, mock_get_max_timestamp_to):
        """Verify small price moves don't create phantom bricks (both trades in level 0)."""
        data_frame = self.get_raw_renko([Decimal("19812.55"), Decimal("19811.55")])
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 0)  # No level exit, no bricks emitted

    def test_body_brick_complete_traversal(self, mock_get_max_timestamp_to):
        """Body brick: entered from below, exited to above."""
        data_frame = self.get_raw_renko([Decimal("100"), Decimal("101"), Decimal("102.02")])
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all().order_by("timestamp")
        self.assertEqual(candle_data.count(), 2)
        brick0 = candle_data[0].json_data
        brick0_renko = candle_data[0].renko_data
        brick1 = candle_data[1].json_data
        brick1_renko = candle_data[1].renko_data
        # Both bricks are bodies (complete traversal)
        self.assertEqual(brick0_renko.level, 0)
        self.assertEqual(brick0_renko.kind, RenkoKind.BODY)
        self.assertEqual(brick1_renko.level, 1)
        self.assertEqual(brick1_renko.kind, RenkoKind.BODY)

    def test_wick_brick_failed_excursion(self, mock_get_max_timestamp_to):
        """Wick brick: pending wick not emitted until parent body emitted."""
        data_frame = self.get_raw_renko([Decimal("100"), Decimal("101"), Decimal("100.5")])
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all().order_by("timestamp")
        # Only 1 brick: level 0 body. Level 1 wick is pending (assigned to level 0).
        # Since we don't exit level 0 again, the wick is never emitted.
        self.assertEqual(candle_data.count(), 1)
        brick0 = candle_data[0].json_data
        brick0_renko = candle_data[0].renko_data
        # Brick 0 is body (seed level 0)
        self.assertEqual(brick0_renko.kind, RenkoKind.BODY)
        self.assertEqual(brick0_renko.level, 0)

    def test_multi_level_jump_creates_empty_bricks(self, mock_get_max_timestamp_to):
        """Multi-level jump from 0 to 3 should create empty bricks for levels 1, 2."""
        data_frame = self.get_raw_renko([Decimal("100"), Decimal("103.04")])
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all().order_by("renko_data__sequence")
        self.assertEqual(len(candle_data), 3)
        # Level 0 has trades, levels 1 and 2 are empty
        self.assertEqual(candle_data[0].renko_data.level, 0)
        self.assertEqual(candle_data[1].renko_data.level, 1)
        self.assertEqual(candle_data[2].renko_data.level, 2)

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
            places=15
        )

    def test_ping_pong_delays_wick_emission(self, mock_get_max_timestamp_to):
        """Rapid ping-pong should not emit wicks until parent body emitted."""
        # 100 → 101 (0→1) → 102.5 (1→2) → 101.5 (2→1) → 102.5 (1→2) → 101.5 (2→1) → 99.5 (1→0→-1)
        # Level 2 boundary = 100 * 1.01^2 = 102.01, so use 102.5 to ensure level 2
        prices = [Decimal("100"), Decimal("101"), Decimal("102.5"),
                  Decimal("101.5"), Decimal("102.5"), Decimal("101.5"), Decimal("99.5")]

        df = self.get_raw_renko(prices)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, df)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)

        candles = CandleData.objects.all().order_by("renko_data__sequence")

        # Should emit:
        # 1. level=0, body, up (100→101)
        # 2. level=1, body, up (101→102)
        # 3. level=2, wick, up (uncompleted excursion, emitted with parent body at level 1)
        # 4. level=1, body, down (parent body triggers wick emission)
        # 5. level=0, body, down (empty)

        self.assertEqual(len(candles), 5)
        self.assertEqual(candles[0].renko_data.level, 0)
        self.assertEqual(candles[0].renko_data.kind, RenkoKind.BODY)
        self.assertEqual(candles[0].renko_data.direction, 1)
        self.assertEqual(candles[1].renko_data.level, 1)
        self.assertEqual(candles[1].renko_data.kind, RenkoKind.BODY)
        self.assertEqual(candles[1].renko_data.direction, 1)
        self.assertEqual(candles[2].renko_data.level, 2)
        self.assertEqual(candles[2].renko_data.kind, RenkoKind.WICK)
        self.assertEqual(candles[2].renko_data.direction, +1)  # Upper wick, side-based
        self.assertEqual(candles[3].renko_data.level, 1)
        self.assertEqual(candles[3].renko_data.kind, RenkoKind.BODY)
        self.assertEqual(candles[3].renko_data.direction, -1)
        self.assertEqual(candles[4].renko_data.level, 0)
        self.assertEqual(candles[4].renko_data.kind, RenkoKind.BODY)
        self.assertEqual(candles[4].renko_data.direction, -1)

    def test_pending_wick_emitted_on_continuation(self, mock_get_max_timestamp_to):
        """Pending wick emitted before body brick even on continuation."""
        # 100 → 101 (0→1) → 102.5 (1→2) → 101.5 (2→1, pending wick) → 102.5 (1→2, re-enter) → 103.5 (2→3, emit wick+body)
        # Level boundaries: L1=101, L2=102.01, L3=103.03
        prices = [Decimal("100"), Decimal("101"), Decimal("102.5"),
                  Decimal("101.5"), Decimal("102.5"), Decimal("103.5")]

        df = self.get_raw_renko(prices)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, df)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)

        candles = CandleData.objects.all().order_by("renko_data__sequence")

        # Should emit:
        # 1. level=0, body, up (100→101)
        # 2. level=1, body, up (101→102.5)
        # 3. level=1, wick, down (the 2→1→2 downward excursion, emitted before level 2 body)
        # 4. level=2, body, up (102.5→103.5, continuation)

        self.assertEqual(len(candles), 4)
        self.assertEqual(candles[0].renko_data.level, 0)
        self.assertEqual(candles[0].renko_data.kind, RenkoKind.BODY)
        self.assertEqual(candles[0].renko_data.direction, 1)
        self.assertEqual(candles[1].renko_data.level, 1)
        self.assertEqual(candles[1].renko_data.kind, RenkoKind.BODY)
        self.assertEqual(candles[1].renko_data.direction, 1)
        self.assertEqual(candles[2].renko_data.level, 1)
        self.assertEqual(candles[2].renko_data.kind, RenkoKind.WICK)
        self.assertEqual(candles[2].renko_data.direction, -1)  # Lower wick, side-based
        self.assertEqual(candles[3].renko_data.level, 2)
        self.assertEqual(candles[3].renko_data.kind, RenkoKind.BODY)
        self.assertEqual(candles[3].renko_data.direction, 1)

    def test_wick_level_invariant_one_away_from_parent(self, mock_get_max_timestamp_to):
        """Verify all wicks are exactly 1 level away from adjacent body brick."""
        test_scenarios = [
            # Scenario 1: Simple reversal with wick (100→101→99)
            {
                "prices": [Decimal("100"), Decimal("101"), Decimal("99")],
                "description": "Simple reversal",
            },
            # Scenario 2: Ping-pong with wicks
            {
                "prices": [
                    Decimal("100"),
                    Decimal("101"),
                    Decimal("102.5"),
                    Decimal("101.5"),
                    Decimal("102.5"),
                    Decimal("101.5"),
                    Decimal("99"),
                ],
                "description": "Ping-pong scenario",
            },
            # Scenario 3: Continuation with pending wick
            {
                "prices": [
                    Decimal("100"),
                    Decimal("101"),
                    Decimal("102.5"),
                    Decimal("101.5"),
                    Decimal("102.5"),
                    Decimal("103.5"),
                ],
                "description": "Continuation with wick",
            },
        ]

        for scenario in test_scenarios:
            with self.subTest(scenario=scenario["description"]):
                # Clear Renko state between scenarios (prevent flaky tests)
                CandleData.objects.filter(candle=self.candle).delete()
                CandleCache.objects.filter(candle=self.candle).delete()
                TradeData.objects.filter(
                    symbol=self.symbol,
                    timestamp__gte=self.timestamp_from,
                    timestamp__lt=self.one_minute_from_now
                ).delete()

                df = self.get_raw_renko(scenario["prices"])
                self.write_trade_data(self.timestamp_from, self.one_minute_from_now, df)
                aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)

                candles = CandleData.objects.all().order_by("renko_data__sequence")

                # Check all wicks satisfy the 1-level invariant
                for i, candle in enumerate(candles):
                    if candle.renko_data.kind == RenkoKind.WICK:
                        # Find adjacent body bricks (previous and next in sequence)
                        # A wick should be 1 level away from at least one adjacent body
                        prev_body = None
                        next_body = None

                        # Look backward for previous body
                        for j in range(i - 1, -1, -1):
                            if candles[j].renko_data.kind == RenkoKind.BODY:
                                prev_body = candles[j]
                                break

                        # Look forward for next body
                        for j in range(i + 1, len(candles)):
                            if candles[j].renko_data.kind == RenkoKind.BODY:
                                next_body = candles[j]
                                break

                        # Wick must be exactly 1 level away from at least one adjacent body
                        wick_level = candle.renko_data.level
                        distances = []

                        if prev_body is not None:
                            distances.append(abs(wick_level - prev_body.renko_data.level))
                        if next_body is not None:
                            distances.append(abs(wick_level - next_body.renko_data.level))

                        # At least one distance should be exactly 1
                        self.assertIn(
                            1,
                            distances,
                            f"Wick at level {wick_level} (sequence {candle.renko_data.sequence}) "
                            f"is not exactly 1 level from any adjacent body. "
                            f"Distances: {distances}, "
                            f"prev_body: {prev_body.renko_data.level if prev_body else None}, "
                            f"next_body: {next_body.renko_data.level if next_body else None}",
                        )

    def test_get_candle_data_canonicalization(self, mock_get_max_timestamp_to):
        """Verify get_candle_data canonicalizes return-to-previous-level pattern."""
        # Clear Renko state before test
        CandleData.objects.filter(candle=self.candle).delete()
        CandleCache.objects.filter(candle=self.candle).delete()
        TradeData.objects.filter(
            symbol=self.symbol,
            timestamp__gte=self.timestamp_from,
            timestamp__lt=self.one_minute_from_now
        ).delete()

        # Create pattern: 1 -> 2 -> 3(wick) -> 2 -> 1
        # Use precise price calculations
        # Level boundaries with target_percentage_change = 0.01:
        # L0=100, L1=101.00, L2=102.01, L3=103.0301
        prices = [
            Decimal("100"),     # level 0
            Decimal("101.5"),   # level 1 (>101.00)
            Decimal("103.5"),   # level 3 (>103.03) - multi-level jump
            Decimal("102.5"),   # level 2 (>102.01, <103.03) - return to 2
            Decimal("100.5"),   # level 0 (<101.00) - reversal
        ]
        data_frame = self.get_raw_renko(prices)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)

        # Query raw DB rows (via ORM)
        raw_candles = CandleData.objects.filter(candle=self.candle).select_related("renko_data").order_by("renko_data__sequence")

        # Verify raw structure
        raw_levels = [c.renko_data.level for c in raw_candles]
        raw_kinds = [c.renko_data.kind for c in raw_candles]
        raw_directions = [c.renko_data.direction for c in raw_candles]

        # The seed brick (first price=100 at level 0) is not emitted until we exit it
        # Expected pattern with emit-on-exit:
        # - Trade at 100 (L0): initializes, no emit
        # - Trade at 101.5 (L1): emits body(0↑), enters L1
        # - Trade at 103.5 (L3): emits body(1↑), body(2↑), wick(3)
        # - Trade at 102.5 (L2): emits body(2↓)
        # - Trade at 100.5 (L0): emits body(1↓), body(0↓)
        # Total: 6 rows
        self.assertEqual(raw_candles.count(), 6, f"Expected 6 raw DB rows, got {raw_candles.count()}")
        self.assertEqual(raw_levels, [0, 1, 2, 3, 2, 1])
        self.assertEqual(raw_kinds, [RenkoKind.BODY, RenkoKind.BODY, RenkoKind.BODY, RenkoKind.WICK, RenkoKind.BODY, RenkoKind.BODY])
        self.assertEqual(raw_directions, [1, 1, 1, 1, -1, -1])

        # Query via get_candle_data(is_complete=False)
        df = self.candle.get_candle_data(
            self.timestamp_from,
            self.one_minute_from_now,
            is_complete=False
        )

        # Verify canonicalization: fewer rows (body(2↓) merged into body(2↑))
        # Expected canonicalized with is_complete=False:
        # body(0↑), body(1↑), body(2↑ merged), wick(3), body(1↓), body(0↓ incomplete from cache)
        self.assertEqual(len(df), 6, "Expected 6 rows (5 complete + 1 incomplete from cache)")

        # Verify wick row is present at level 3
        wick_rows = df[df["renko_kind"] == RenkoKind.WICK]
        self.assertEqual(len(wick_rows), 1, "Expected 1 wick row")
        self.assertEqual(wick_rows.iloc[0]["renko_level"], 3)

        # Find the level-2 body row (should be merged)
        level_2_bodies = df[
            (df["renko_kind"] == RenkoKind.BODY) & (df["renko_level"] == 2)
        ]
        self.assertEqual(len(level_2_bodies), 1, "Expected 1 level-2 body row (merged)")

        level_2_body = level_2_bodies.iloc[0]
        # Verify direction unchanged (not flipped)
        self.assertEqual(level_2_body["renko_direction"], 1, "Level-2 direction should remain 1 (up)")
        # Verify timestamp_end is set
        self.assertIn("timestamp_end", level_2_body, "Merged row should have timestamp_end")
        self.assertIsNotNone(level_2_body["timestamp_end"])
        # Verify renko_sequence_end is set
        self.assertIn("renko_sequence_end", level_2_body, "Merged row should have renko_sequence_end")
        self.assertIsNotNone(level_2_body["renko_sequence_end"])
        # Note: timestamp and timestamp_end may be equal in test due to same base timestamp

        # Verify NO direction flip at "return to 2"
        # The body stream should show: 0(↑), 1(↑), 2(↑ merged), 1(↓), 0(↓ incomplete)
        body_df = df[df["renko_kind"] == RenkoKind.BODY]
        body_levels = body_df["renko_level"].tolist()
        body_directions = body_df["renko_direction"].tolist()
        self.assertEqual(body_levels, [0, 1, 2, 1, 0], "Body levels after canonicalization with incomplete")
        self.assertEqual(body_directions, [1, 1, 1, -1, -1], "Body directions: no flip at level 2")

        # Test is_complete parameter
        df_complete = self.candle.get_candle_data(
            self.timestamp_from,
            self.one_minute_from_now,
            is_complete=True
        )
        # Should exclude last body brick (level 1↓)
        # Expected: body(0↑), body(1↑), body(2↑ merged), wick(3)
        self.assertEqual(len(df_complete), 4, "is_complete=True should exclude last body")
        # Verify last row is wick, not body
        self.assertEqual(
            df_complete.iloc[-1]["renko_kind"],
            RenkoKind.WICK,
            "Last row should be wick after excluding last body",
        )
        # Verify last body is at level 2 (not level 1)
        last_body_complete = df_complete[df_complete["renko_kind"] == RenkoKind.BODY].iloc[
            -1
        ]
        self.assertEqual(last_body_complete["renko_level"], 2, "Last body should be level 2 (merged)")

    def test_get_candle_data_incomplete_brick_from_cache(self, mock_get_max_timestamp_to):
        """Verify get_candle_data synthesizes incomplete brick from CandleCache."""
        # Clear state
        CandleData.objects.filter(candle=self.candle).delete()
        CandleCache.objects.filter(candle=self.candle).delete()
        TradeData.objects.filter(
            symbol=self.symbol,
            timestamp__gte=self.timestamp_from,
            timestamp__lt=self.one_minute_from_now
        ).delete()

        # Create scenario with incomplete brick in cache
        # Prices: 100 (L0) -> 101.5 (L1)
        # Level 0 emitted to DB, level 1 remains in cache (incomplete)
        prices = [
            Decimal("100"),     # level 0 - initializes
            Decimal("101.5"),   # level 1 - emits body(0↑), enters L1 but doesn't exit
        ]
        data_frame = self.get_raw_renko(prices)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(self.candle, self.timestamp_from, self.one_minute_from_now)

        # Verify DB contains only level 0 body
        db_candles = CandleData.objects.filter(candle=self.candle).order_by("renko_data__sequence")
        self.assertEqual(db_candles.count(), 1)
        self.assertEqual(db_candles[0].renko_data.level, 0)
        self.assertEqual(db_candles[0].renko_data.kind, RenkoKind.BODY)

        # Verify cache contains level 1 state
        cache_obj = CandleCache.objects.filter(candle=self.candle).order_by("-timestamp").first()
        self.assertIsNotNone(cache_obj)
        cache = cache_obj.json_data
        self.assertEqual(cache["active_level"], 1)

        # Query with is_complete=False (should include incomplete brick)
        df_incomplete = self.candle.get_candle_data(
            self.timestamp_from,
            self.one_minute_from_now,
            is_complete=False
        )

        # Verify output contains both DB brick and synthetic incomplete brick
        self.assertEqual(len(df_incomplete), 2, "Expected 2 rows: body(0↑) from DB + incomplete body(1↑) from cache")

        # Verify first row is body(0↑) from DB
        row0 = df_incomplete.iloc[0]
        self.assertEqual(row0["renko_level"], 0)
        self.assertEqual(row0["renko_kind"], RenkoKind.BODY)
        self.assertEqual(row0["renko_direction"], 1)

        # Verify second row is incomplete body(1↑) from cache
        row1 = df_incomplete.iloc[1]
        self.assertEqual(row1["renko_level"], 1)
        self.assertEqual(row1["renko_kind"], RenkoKind.BODY)
        self.assertEqual(row1["renko_direction"], 1, "Direction should be 1 (from_below)")
        self.assertEqual(row1["renko_sequence"], cache["brick_sequence"], "Sequence should match cache")

        # Verify OHLCV fields are present
        for field in ["open", "high", "low", "close"]:
            self.assertIn(field, row1, f"Incomplete brick should have {field}")
            self.assertIsNotNone(row1[field])

        # Query with is_complete=True (should exclude incomplete brick)
        df_complete = self.candle.get_candle_data(
            self.timestamp_from,
            self.one_minute_from_now,
            is_complete=True
        )

        # Verify incomplete brick is excluded (only DB bricks, minus last body)
        # In this case, level 0 is the last body, so it gets excluded too
        self.assertEqual(len(df_complete), 0, "is_complete=True should exclude all incomplete bricks")

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
            self.assertEqual(ratio.quantize(Decimal("0.0001")), (Decimal(1) + p).quantize(Decimal("0.0001")))

    def test_negative_levels_below_origin(self, mock_get_max_timestamp_to):
        """Verify negative levels when price < origin_price."""
        # Use origin_price = 1, first price = 0.5 (below origin)
        candle = RenkoBrick.objects.create(
            json_data={
                "source_data": FileData.RAW,
                "target_percentage_change": Decimal("0.01"),
                "origin_price": Decimal("1"),
            }
        )
        candle.symbols.add(self.symbol)

        # Price 0.5 should map to negative level
        # level = floor(ln(0.5/1) / ln(1.01)) = floor(ln(0.5) / 0.00995) ≈ floor(-69.66) = -70
        data_frame = pd.DataFrame([
            self.get_random_trade(timestamp=self.timestamp_from, price=Decimal("0.5"))
        ])
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame)
        aggregate_candles(candle, self.timestamp_from, self.one_minute_from_now)

        # Verify cache has negative active_level
        cache = CandleCache.objects.filter(candle=candle).first()
        self.assertIsNotNone(cache)
        self.assertLess(cache.json_data["active_level"], 0)

    def test_deterministic_levels_across_series(self, mock_get_max_timestamp_to):
        """Verify same price maps to same level across different series."""
        origin = Decimal("1")
        p = Decimal("0.01")

        # Create two candles with same config
        candle1 = RenkoBrick.objects.create(
            json_data={
                "source_data": FileData.RAW,
                "target_percentage_change": p,
                "origin_price": origin,
            }
        )
        candle1.symbols.add(self.symbol)
        candle2 = RenkoBrick.objects.create(
            json_data={
                "source_data": FileData.RAW,
                "target_percentage_change": p,
                "origin_price": origin,
            }
        )
        candle2.symbols.add(self.symbol)

        # Same price for both
        price = Decimal("100")

        # Process price for candle1
        data_frame1 = pd.DataFrame([
            self.get_random_trade(timestamp=self.timestamp_from, price=price)
        ])
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame1)
        aggregate_candles(candle1, self.timestamp_from, self.one_minute_from_now)

        # Process price for candle2
        data_frame2 = pd.DataFrame([
            self.get_random_trade(timestamp=self.timestamp_from, price=price)
        ])
        TradeData.objects.filter(symbol=self.symbol).delete()  # Clear trades
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, data_frame2)
        aggregate_candles(candle2, self.timestamp_from, self.one_minute_from_now)

        # Verify both have same active_level
        cache1 = CandleCache.objects.filter(candle=candle1).first()
        cache2 = CandleCache.objects.filter(candle=candle2).first()
        self.assertEqual(cache1.json_data["active_level"], cache2.json_data["active_level"])
