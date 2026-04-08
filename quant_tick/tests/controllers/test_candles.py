from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import time_machine
from django.test import TestCase
from pandas import DataFrame

from quant_tick.constants import FileData, Frequency, SampleType
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
    ConstantCandle,
    TimeBasedCandle,
    TradeData,
)
from quant_tick.models.candles import camel_to_snake

from ..base import BaseSymbolTest, BaseWriteTradeDataTest


class BaseMinuteIteratorTest:

    def setUp(self) -> None:
        super().setUp()
        self.timestamp_from = get_min_time(get_current_time(), "1d")
        self.one_minute_from_now = self.timestamp_from + pd.Timedelta("1min")
        self.two_minutes_from_now = self.timestamp_from + pd.Timedelta("2min")
        self.three_minutes_from_now = self.timestamp_from + pd.Timedelta("3min")


class BaseHourIteratorTest:

    def setUp(self) -> None:
        super().setUp()
        self.timestamp_from = get_min_time(get_current_time(), "1d")
        self.one_hour_from_now = self.timestamp_from + pd.Timedelta("1h")
        self.two_hours_from_now = self.timestamp_from + pd.Timedelta("2h")
        self.three_hours_from_now = self.timestamp_from + pd.Timedelta("3h")


class BaseDayIteratorTest:

    def setUp(self) -> None:
        super().setUp()
        self.timestamp_from = get_min_time(get_current_time(), "1d")
        self.one_day_from_now = self.timestamp_from + pd.Timedelta("1d")
        self.two_days_from_now = self.timestamp_from + pd.Timedelta("2d")
        self.three_days_from_now = self.timestamp_from + pd.Timedelta("3d")


class BaseCandleCacheIteratorTest(BaseSymbolTest):

    def setUp(self) -> None:
        super().setUp()
        self.symbol = self.get_symbol()
        self.candle = self.get_candle()

    def get_candle(self) -> Candle:
        return Candle.objects.create(
            symbol=self.symbol, json_data={"source_data": FileData.RAW}
        )

    def get_values(self) -> list[tuple[datetime, datetime]]:
        return [
            (ts_from, ts_to)
            for ts_from, ts_to, _ in self.candle.iter_all(
                self.timestamp_from, self.timestamp_to
            )
        ]


@time_machine.travel(datetime(2009, 1, 4), tick=False)
class CandleTest(BaseSymbolTest, BaseDayIteratorTest, TestCase):

    def setUp(self):
        super().setUp()
        self.candle = Candle.objects.create(
            symbol=self.get_symbol(), json_data={"source_data": FileData.RAW}
        )

    def create_candle_cache(self, timestamp: datetime) -> CandleCache:
        return CandleCache.objects.create(
            candle=self.candle, timestamp=timestamp, frequency=Frequency.DAY
        )

    def test_initial_timestamp_from_without_candle_date_from(self):
        timestamp_from, timestamp_to, _ = self.candle.initialize(
            self.timestamp_from, self.three_days_from_now
        )
        self.assertEqual(timestamp_from, self.timestamp_from)
        self.assertEqual(timestamp_to, self.three_days_from_now)

    def test_initial_timestamp_from_with_candle_date_from(self):
        self.candle.date_from = self.one_day_from_now.date()
        timestamp_from, timestamp_to, _ = self.candle.initialize(
            self.timestamp_from, self.three_days_from_now
        )
        self.assertEqual(timestamp_from, self.one_day_from_now)

    def test_initial_timestamp_from_with_symbol_date_from(self):
        symbol = self.candle.symbol
        symbol.date_from = self.one_day_from_now.date()
        symbol.save()
        timestamp_from, timestamp_to, _ = self.candle.initialize(
            self.timestamp_from, self.three_days_from_now
        )
        self.assertEqual(timestamp_from, self.one_day_from_now)
        self.assertEqual(timestamp_to, self.three_days_from_now)

    def test_initial_timestamp_from_uses_later_symbol_or_candle_date_from(self):
        symbol = self.candle.symbol
        symbol.date_from = self.two_days_from_now.date()
        symbol.save()
        self.candle.date_from = self.one_day_from_now.date()
        timestamp_from, timestamp_to, _ = self.candle.initialize(
            self.timestamp_from, self.three_days_from_now
        )
        self.assertEqual(timestamp_from, self.two_days_from_now)
        self.assertEqual(timestamp_to, self.three_days_from_now)

    def test_initial_timestamp_from_skips_before_symbol_date_from(self):
        symbol = self.candle.symbol
        symbol.date_from = self.three_days_from_now.date()
        symbol.save()
        timestamp_from, timestamp_to, data = self.candle.initialize(
            self.timestamp_from, self.two_days_from_now
        )
        self.assertEqual(timestamp_from, self.two_days_from_now)
        self.assertEqual(timestamp_to, self.two_days_from_now)
        self.assertEqual(data, {})

    def test_initial_timestamp_from_with_candle_cache(self):
        for i in range(2):
            self.create_candle_cache(self.timestamp_from + pd.Timedelta(f"{i}d"))
        timestamp_from, timestamp_to, _ = self.candle.initialize(
            self.timestamp_from, self.three_days_from_now
        )
        self.assertEqual(timestamp_from, self.two_days_from_now)
        self.assertEqual(timestamp_to, self.three_days_from_now)

    def test_initial_timestamp_from_with_candle_cache_and_retry(self):
        for i in range(2):
            self.create_candle_cache(self.timestamp_from + pd.Timedelta(f"{i}d"))
        timestamp_from, timestamp_to, _ = self.candle.initialize(
            self.timestamp_from, self.three_days_from_now, retry=True
        )
        self.assertEqual(timestamp_from, self.timestamp_from)
        self.assertEqual(timestamp_to, self.three_days_from_now)

    def test_initial_timestamp_from_with_both_candle_date_from_and_candle_cache(self):
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
    "quant_tick.models.candles.get_current_time",
    return_value=datetime(2009, 1, 4, 0, 3).replace(tzinfo=UTC),
)
class TimeBasedMinuteFrequencyCandleTest(
    BaseMinuteIteratorTest,
    BaseWriteTradeDataTest,
    BaseCandleCacheIteratorTest,
    TestCase,
):

    def get_candle(self) -> Candle:
        return TimeBasedCandle.objects.create(
            symbol=self.symbol,
            json_data={"source_data": FileData.RAW, "window": "1min"},
        )

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, pd.DataFrame([])
        )

    def test_one_candle_from_trade_in_the_first_minute(self, mock_get_max_timestamp_to):
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, filtered)
        self.candle.candles(self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_one_candle_from_trade_in_the_first_minute_with_retry(
        self, mock_get_max_timestamp_to
    ):
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, filtered)
        for i in range(2):
            self.candle.candles(
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
        filtered_1 = self.get_filtered(
            self.timestamp_from,
            price=Decimal("5"),
            notional=Decimal("1"),
            tick_rule=1,
        )
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, filtered_1)
        filtered_2 = self.get_filtered(
            self.one_minute_from_now,
            price=Decimal("6"),
            notional=Decimal("2"),
            tick_rule=-1,
        )
        self.write_trade_data(
            self.one_minute_from_now, self.two_minutes_from_now, filtered_2
        )
        self.candle.candles(self.timestamp_from, self.two_minutes_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        self.assertEqual(candle_data[1].timestamp, self.one_minute_from_now)

    def test_two_candles_from_trades_in_first_and_third_minute(
        self, mock_get_max_timestamp_to
    ):
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
        self.candle.candles(self.timestamp_from, self.three_minutes_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        self.assertEqual(candle_data[1].timestamp, self.two_minutes_from_now)

    def test_gap_logs_warning_and_stops_iteration(self, mock_get_max_timestamp_to):
        filtered_1 = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, filtered_1)
        filtered_2 = self.get_filtered(self.two_minutes_from_now)
        self.write_trade_data(
            self.two_minutes_from_now, self.three_minutes_from_now, filtered_2
        )

        with self.assertLogs("quant_tick.models.candles", level="WARNING") as logs:
            values = list(
                self.candle.iter_all(self.timestamp_from, self.three_minutes_from_now)
            )

        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertIn("stopped on TradeData gap", logs.output[0])
        self.assertIn(str(self.candle), logs.output[0])


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.models.candles.get_current_time",
    return_value=datetime(2009, 1, 4, 0, 3).replace(tzinfo=UTC),
)
class TimeBasedTwoMinuteFrequencyCandleTest(
    BaseMinuteIteratorTest,
    BaseWriteTradeDataTest,
    BaseCandleCacheIteratorTest,
    TestCase,
):

    def get_candle(self) -> Candle:
        return TimeBasedCandle.objects.create(
            symbol=self.symbol,
            json_data={"source_data": FileData.RAW, "window": "2min"},
        )

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, pd.DataFrame([])
        )

    def test_next_cache_created_if_candle_window_exceeded(
        self, mock_get_max_timestamp_to
    ):
        filtered_1 = self.get_filtered(
            self.timestamp_from,
            price=Decimal("5"),
            notional=Decimal("1"),
            tick_rule=1,
        )
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, filtered_1)
        filtered_2 = self.get_filtered(
            self.one_minute_from_now,
            price=Decimal("6"),
            notional=Decimal("2"),
            tick_rule=-1,
        )
        self.write_trade_data(
            self.one_minute_from_now, self.two_minutes_from_now, filtered_2
        )
        self.candle.candles(self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all()
        self.assertFalse(candle_data.exists())
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)

    def test_one_candle_from_one_trade_in_the_first_minute_and_another_in_the_second(
        self, mock_get_max_timestamp_to
    ):
        filtered_1 = self.get_filtered(
            self.timestamp_from,
            price=Decimal("5"),
            notional=Decimal("1"),
            tick_rule=1,
        )
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, filtered_1)
        filtered_2 = self.get_filtered(
            self.one_minute_from_now,
            price=Decimal("6"),
            notional=Decimal("2"),
            tick_rule=-1,
        )
        self.write_trade_data(
            self.one_minute_from_now, self.two_minutes_from_now, filtered_2
        )
        for i in range(2):
            self.candle.candles(
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
        candle = {camel_to_snake(key): value for key, value in candle.items()}
        actual = next(self.candle.get_candle_data())
        for key in ("timestamp",):
            del actual[key]
        self.assertGreater(actual["realized_variance"], 0)
        self.assertGreater(candle["realized_variance"], 0)
        self.assertAlmostEqual(
            float(actual.pop("realized_variance")),
            float(candle.pop("realized_variance")),
            places=12,
        )
        self.assertEqual(actual, candle)


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.models.candles.get_current_time",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=UTC),
)
class TimeBasedHourFrequencyCandleTest(
    BaseHourIteratorTest, BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):

    def get_candle(self) -> Candle:
        return TimeBasedCandle.objects.create(
            symbol=self.symbol,
            json_data={"source_data": FileData.RAW, "window": "1h"},
        )

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, pd.DataFrame([])
        )

    def test_one_candle_from_trade_in_the_first_hour(self, mock_get_max_timestamp_to):
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        self.candle.candles(self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_one_candle_from_trade_in_the_first_hour_with_retry(
        self, mock_get_max_timestamp_to
    ):
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        for i in range(2):
            self.candle.candles(
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
        filtered_1 = self.get_filtered(
            self.timestamp_from,
            price=Decimal("5"),
            notional=Decimal("1"),
            tick_rule=1,
        )
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        filtered_2 = self.get_filtered(
            self.one_hour_from_now,
            price=Decimal("6"),
            notional=Decimal("2"),
            tick_rule=-1,
        )
        self.write_trade_data(
            self.one_hour_from_now, self.two_hours_from_now, filtered_2
        )
        self.candle.candles(self.timestamp_from, self.two_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        self.assertEqual(candle_data[1].timestamp, self.one_hour_from_now)

    def test_two_candles_from_trades_in_first_and_third_hour(
        self, mock_get_max_timestamp_to
    ):
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
        self.candle.candles(self.timestamp_from, self.three_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        self.assertEqual(candle_data[1].timestamp, self.two_hours_from_now)

    def test_candle_cache_created_from_trade_in_the_first_minute(
        self, mock_get_max_timestamp_to
    ):
        filtered = self.get_filtered(self.timestamp_from)
        one_minute_from_now = self.timestamp_from + pd.Timedelta("1min")
        self.write_trade_data(self.timestamp_from, one_minute_from_now, filtered)
        self.candle.candles(self.timestamp_from, one_minute_from_now)
        candle_data = CandleData.objects.all()
        self.assertFalse(candle_data.exists())
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)
        candle = aggregate_candle(filtered)
        self.assertEqual(candle_cache.first().json_data["next"], candle)

    def test_one_candle_from_trade_with_existing_one_minute_candle_cache(
        self, mock_get_max_timestamp_to
    ):
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
        self.candle.candles(self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        last_candle_cache = CandleCache.objects.last()
        self.assertEqual(last_candle_cache.json_data, {})


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.models.candles.get_current_time",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=UTC),
)
class TimeBasedTwoHourFrequencyCandleTest(
    BaseHourIteratorTest, BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):

    def get_candle(self) -> Candle:
        return TimeBasedCandle.objects.create(
            symbol=self.symbol,
            json_data={"source_data": FileData.RAW, "window": "2h"},
        )

    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, pd.DataFrame([])
        )

    def test_next_cache_created_if_candle_window_exceeded(
        self, mock_get_max_timestamp_to
    ):
        filtered_1 = self.get_filtered(
            self.timestamp_from,
            price=Decimal("5"),
            notional=Decimal("1"),
            tick_rule=1,
        )
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        filtered_2 = self.get_filtered(
            self.one_hour_from_now,
            price=Decimal("6"),
            notional=Decimal("2"),
            tick_rule=-1,
        )
        self.write_trade_data(
            self.one_hour_from_now, self.two_hours_from_now, filtered_2
        )
        self.candle.candles(self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertFalse(candle_data.exists())
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)

    def test_one_candle_from_one_trade_in_the_first_hour_and_another_in_the_second(
        self, mock_get_max_timestamp_to
    ):
        filtered_1 = self.get_filtered(
            self.timestamp_from,
            price=Decimal("5"),
            notional=Decimal("1"),
            tick_rule=1,
        )
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        filtered_2 = self.get_filtered(
            self.one_hour_from_now,
            price=Decimal("6"),
            notional=Decimal("2"),
            tick_rule=-1,
        )
        self.write_trade_data(
            self.one_hour_from_now, self.two_hours_from_now, filtered_2
        )
        for i in range(2):
            self.candle.candles(
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
        candle = {camel_to_snake(key): value for key, value in candle.items()}
        actual = next(self.candle.get_candle_data())
        for key in ("timestamp",):
            del actual[key]
        self.assertGreater(actual["realized_variance"], 0)
        self.assertGreater(candle["realized_variance"], 0)
        self.assertAlmostEqual(
            float(actual.pop("realized_variance")),
            float(candle.pop("realized_variance")),
            places=12,
        )
        self.assertEqual(actual, candle)


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.models.candles.get_current_time",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=UTC),
)
class ConstantNotionalHourFrequencyCandleTest(
    BaseHourIteratorTest, BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):

    def get_candle(self) -> Candle:
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
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, pd.DataFrame([])
        )

    def test_no_candles_from_trade_in_the_first_hour(self, mock_get_max_timestamp_to):
        expected = Decimal("0.5")
        filtered = self.get_filtered(self.timestamp_from, notional=expected)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        self.candle.candles(self.timestamp_from, self.one_hour_from_now)
        self.assertFalse(CandleData.objects.exists())
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)
        self.assertEqual(candle_cache[0].json_data["sample_value"], expected)

    def test_one_candle_from_trade_in_the_first_hour(self, mock_get_max_timestamp_to):
        filtered = self.get_filtered(self.timestamp_from, notional=Decimal("1"))
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        self.candle.candles(self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_one_candle_from_trade_in_the_first_hour_with_retry(
        self, mock_get_max_timestamp_to
    ):
        filtered = self.get_filtered(self.timestamp_from, notional=Decimal("1"))
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        for i in range(2):
            self.candle.candles(
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
        filtered = self.get_filtered(self.timestamp_from, notional=Decimal("1"))
        for i in range(2):
            retry = bool(i)
            if retry:
                missing_trade = self.get_filtered(
                    self.timestamp_from, notional=Decimal("0.5")
                )
                filtered = pd.concat([filtered, missing_trade])
            self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
            self.candle.candles(self.timestamp_from, self.one_hour_from_now, retry=True)
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
        filtered_1 = self.get_filtered(self.timestamp_from, notional=Decimal("0.5"))
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        self.candle.candles(self.timestamp_from, self.one_hour_from_now)
        filtered_2 = self.get_filtered(self.one_hour_from_now, notional=Decimal("0.5"))
        self.write_trade_data(
            self.one_hour_from_now, self.two_hours_from_now, filtered_2
        )
        self.candle.candles(self.one_hour_from_now, self.two_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        trade_data = TradeData.objects.filter(
            symbol=self.symbol,
            timestamp__gte=self.timestamp_from,
            timestamp__lt=self.two_hours_from_now,
        ).order_by("timestamp")
        total_notional = sum(
            self.candle.get_data_frame(
                self.timestamp_from, self.two_hours_from_now, td
            )["totalNotional"].sum()
            for td in trade_data
        )
        self.assertEqual(candle_data[0].notional, total_notional)
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 2)
        self.assertIn("next", candle_cache[0].json_data)
        self.assertNotIn("next", candle_cache[1].json_data)

    def test_two_candles_from_trades_in_the_first_and_second_hour(
        self, mock_get_max_timestamp_to
    ):
        filtered_1 = self.get_filtered(self.timestamp_from, notional=Decimal("1"))
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        filtered_2 = self.get_filtered(self.one_hour_from_now, notional=Decimal("1"))
        self.write_trade_data(
            self.one_hour_from_now, self.two_hours_from_now, filtered_2
        )
        self.candle.candles(self.timestamp_from, self.two_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        self.assertEqual(candle_data[1].timestamp, self.one_hour_from_now)

    def test_two_candles_from_trades_in_first_and_third_hour(
        self, mock_get_max_timestamp_to
    ):
        self.write_trade_data(
            self.timestamp_from,
            self.one_hour_from_now,
            self.get_filtered(self.timestamp_from, notional=Decimal("1")),
        )
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.one_hour_from_now,
            frequency=Frequency.HOUR,
        )
        self.write_trade_data(
            self.two_hours_from_now,
            self.three_hours_from_now,
            self.get_filtered(self.two_hours_from_now, notional=Decimal("1")),
        )
        self.candle.candles(self.timestamp_from, self.three_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        self.assertEqual(candle_data[1].timestamp, self.two_hours_from_now)

    def test_no_candles_without_prior_cache(self, mock_get_max_timestamp_to):
        CandleCache.objects.create(
            candle=self.candle, timestamp=self.timestamp_from, frequency=Frequency.HOUR
        )
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.two_hours_from_now,
            frequency=Frequency.HOUR,
        )
        self.candle.candles(self.two_hours_from_now, self.three_hours_from_now)
        self.assertFalse(CandleData.objects.exists())
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.models.candles.get_current_time",
    return_value=datetime(2009, 1, 5).replace(tzinfo=UTC),
)
class ConstantNotionalDayFrequencyIrregularCandleTest(
    BaseDayIteratorTest, BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):

    def get_candle(self) -> Candle:
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
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, pd.DataFrame([])
        )

    def test_one_incomplete_candle(self, mock_get_max_timestamp_to):
        last_hour = self.timestamp_from + pd.Timedelta("23h")
        filtered = self.get_filtered(last_hour, notional=Decimal("0.5"))
        self.write_trade_data(last_hour, self.one_day_from_now, filtered)
        self.candle.candles(last_hour, self.one_day_from_now)
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, last_hour)
        self.assertTrue(candle_data[0].incomplete)


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.models.candles.get_current_time",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=UTC),
)
class AdaptiveNotionalCandleTest(
    BaseHourIteratorTest, BaseWriteTradeDataTest, BaseCandleCacheIteratorTest, TestCase
):

    def setUp(self):
        super().setUp()
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=get_min_time(self.timestamp_from, value="1d")
            - pd.Timedelta("1d"),
            frequency=Frequency.DAY,
            json_data={"candle": {"notional": 1}},
        )

    def get_candle(self) -> Candle:
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
        TradeData.write(
            self.symbol, timestamp_from, timestamp_to, data_frame, pd.DataFrame([])
        )

    def test_no_candles_from_trade_in_the_first_hour(self, mock_get_max_timestamp_to):
        expected = Decimal("0.5")
        filtered = self.get_filtered(self.timestamp_from, notional=expected)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        self.candle.candles(self.timestamp_from, self.one_hour_from_now)
        self.assertFalse(CandleData.objects.exists())
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)
        self.assertEqual(candle_cache[0].json_data["sample_value"], expected)

    def test_one_candle_from_trade_in_the_first_hour(self, mock_get_max_timestamp_to):
        filtered = self.get_filtered(self.timestamp_from, notional=Decimal("1"))
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        self.candle.candles(self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_one_candle_from_trade_in_the_first_hour_with_retry(
        self, mock_get_max_timestamp_to
    ):
        filtered = self.get_filtered(self.timestamp_from, notional=Decimal("1"))
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        for i in range(2):
            self.candle.candles(
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
        filtered = self.get_filtered(self.timestamp_from, notional=Decimal("1"))
        for i in range(2):
            retry = bool(i)
            if retry:
                missing_trade = self.get_filtered(
                    self.timestamp_from, notional=Decimal("0.5")
                )
                filtered = pd.concat([filtered, missing_trade])
            self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
            self.candle.candles(self.timestamp_from, self.one_hour_from_now, retry=True)
        self.assertEqual(TradeData.objects.count(), 2)
        self.assertEqual(CandleCache.objects.count(), 1)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_one_candle_from_trade_in_the_first_and_second_hour(
        self, mock_get_max_timestamp_to
    ):
        filtered_1 = self.get_filtered(self.timestamp_from, notional=Decimal("0.5"))
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        self.candle.candles(self.timestamp_from, self.one_hour_from_now)
        filtered_2 = self.get_filtered(self.one_hour_from_now, notional=Decimal("0.5"))
        self.write_trade_data(
            self.one_hour_from_now, self.two_hours_from_now, filtered_2
        )
        self.candle.candles(self.one_hour_from_now, self.two_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        trade_data = TradeData.objects.filter(
            symbol=self.symbol,
            timestamp__gte=self.timestamp_from,
            timestamp__lt=self.two_hours_from_now,
        ).order_by("timestamp")
        total_notional = sum(
            self.candle.get_data_frame(
                self.timestamp_from, self.two_hours_from_now, td
            )["totalNotional"].sum()
            for td in trade_data
        )
        self.assertEqual(candle_data[0].notional, total_notional)
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 2)
        self.assertIn("next", candle_cache[0].json_data)
        self.assertNotIn("next", candle_cache[1].json_data)

    def test_two_candles_from_trades_in_the_first_and_second_hour(
        self, mock_get_max_timestamp_to
    ):
        filtered_1 = self.get_filtered(self.timestamp_from, notional=Decimal("1"))
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        filtered_2 = self.get_filtered(self.one_hour_from_now, notional=Decimal("1"))
        self.write_trade_data(
            self.one_hour_from_now, self.two_hours_from_now, filtered_2
        )
        self.candle.candles(self.timestamp_from, self.two_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        self.assertEqual(candle_data[1].timestamp, self.one_hour_from_now)

    def test_two_candles_from_trades_in_first_and_third_hour(
        self, mock_get_max_timestamp_to
    ):
        self.write_trade_data(
            self.timestamp_from,
            self.one_hour_from_now,
            self.get_filtered(self.timestamp_from, notional=Decimal("1")),
        )
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.one_hour_from_now,
            frequency=Frequency.HOUR,
        )
        self.write_trade_data(
            self.two_hours_from_now,
            self.three_hours_from_now,
            self.get_filtered(self.two_hours_from_now, notional=Decimal("1")),
        )
        self.candle.candles(self.timestamp_from, self.three_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        self.assertEqual(candle_data[1].timestamp, self.two_hours_from_now)
