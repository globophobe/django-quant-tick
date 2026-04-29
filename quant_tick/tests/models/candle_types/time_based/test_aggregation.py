from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import time_machine
from django.test import TestCase

from quant_tick.constants import Frequency
from quant_tick.lib import aggregate_candle, get_next_cache
from quant_tick.models import CandleCache, CandleData, TradeData

from ..base import BaseHourIteratorTest, BaseMinuteIteratorTest
from .base import BaseTimeBasedCandleTest


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.models.candles.get_current_time",
    return_value=datetime(2009, 1, 4, 0, 3).replace(tzinfo=UTC),
)
class TimeBasedMinuteFrequencyCandleTest(
    BaseMinuteIteratorTest,
    BaseTimeBasedCandleTest,
    TestCase,
):
    window = "1min"

    def test_one_candle_from_trade_in_the_first_minute(self, mock_get_current_time):
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, filtered)
        self.candle.candles(self.timestamp_from, self.one_minute_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_one_candle_from_trade_in_the_first_minute_with_retry(
        self, mock_get_current_time
    ):
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, filtered)
        for i in range(2):
            self.candle.candles(
                self.timestamp_from,
                self.one_minute_from_now,
                retry=bool(i),
            )
        self.assertEqual(CandleCache.objects.count(), 1)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_two_candles_from_trades_in_the_first_and_second_minute(
        self, mock_get_current_time
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
        self, mock_get_current_time
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

    def test_gap_logs_warning_and_stops_iteration(self, mock_get_current_time):
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

    def test_iter_all_resumes_inside_compacted_trade_data(self, mock_get_current_time):
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp_from,
            frequency=Frequency.DAY,
            raw_data=TradeData.prepare_data(
                self.get_raw(self.timestamp_from + pd.Timedelta("1min"))
            ),
            ok=True,
        )
        timestamp_from = self.timestamp_from + pd.Timedelta("1min")
        timestamp_to = self.timestamp_from + pd.Timedelta("3min")

        values = list(self.candle.iter_all(timestamp_from, timestamp_to))

        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], timestamp_from)
        self.assertEqual(values[0][1], timestamp_to)


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.models.candles.get_current_time",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=UTC),
)
class TimeBasedHourFrequencyCandleTest(
    BaseHourIteratorTest,
    BaseTimeBasedCandleTest,
    TestCase,
):
    window = "1h"

    def test_one_candle_from_trade_in_the_first_hour(self, mock_get_current_time):
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        self.candle.candles(self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_one_candle_from_trade_in_the_first_hour_with_retry(
        self, mock_get_current_time
    ):
        filtered = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        for i in range(2):
            self.candle.candles(
                self.timestamp_from,
                self.one_hour_from_now,
                retry=bool(i),
            )
        self.assertEqual(CandleCache.objects.count(), 1)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_two_candles_from_trades_in_the_first_and_second_hour(
        self, mock_get_current_time
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
        self, mock_get_current_time
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
        self, mock_get_current_time
    ):
        filtered = self.get_filtered(self.timestamp_from)
        one_minute_from_now = self.timestamp_from + pd.Timedelta("1min")
        self.write_trade_data(self.timestamp_from, one_minute_from_now, filtered)
        self.candle.candles(self.timestamp_from, one_minute_from_now)
        self.assertFalse(CandleData.objects.exists())
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)
        candle = aggregate_candle(filtered)
        self.assertEqual(candle_cache.first().json_data["next"], candle)

    def test_one_candle_from_trade_with_existing_one_minute_candle_cache(
        self, mock_get_current_time
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
        self.assertEqual(CandleCache.objects.last().json_data, {})
