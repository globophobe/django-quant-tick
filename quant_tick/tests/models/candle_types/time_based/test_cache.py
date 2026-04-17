from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import time_machine
from django.test import TestCase

from quant_tick.models import CandleCache, CandleData

from ..base import BaseHourIteratorTest, BaseMinuteIteratorTest
from .base import BaseTimeBasedCandleTest


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.models.candles.get_current_time",
    return_value=datetime(2009, 1, 4, 0, 3).replace(tzinfo=UTC),
)
class TimeBasedTwoMinuteFrequencyCandleTest(
    BaseMinuteIteratorTest,
    BaseTimeBasedCandleTest,
    TestCase,
):
    window = "2min"

    def test_next_cache_created_if_candle_window_exceeded(
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
        self.candle.candles(self.timestamp_from, self.one_minute_from_now)
        self.assertFalse(CandleData.objects.exists())
        self.assertEqual(CandleCache.objects.count(), 1)

    def test_one_candle_from_one_trade_in_the_first_minute_and_another_in_the_second(
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
        self.assertEqual(candle_data.first().timestamp, self.timestamp_from)
        self.assert_combined_candle([filtered_1, filtered_2])


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.models.candles.get_current_time",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=UTC),
)
class TimeBasedTwoHourFrequencyCandleTest(
    BaseHourIteratorTest,
    BaseTimeBasedCandleTest,
    TestCase,
):
    window = "2h"

    def test_next_cache_created_if_candle_window_exceeded(
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
        self.candle.candles(self.timestamp_from, self.one_hour_from_now)
        self.assertFalse(CandleData.objects.exists())
        self.assertEqual(CandleCache.objects.count(), 1)

    def test_one_candle_from_one_trade_in_the_first_hour_and_another_in_the_second(
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
        self.assertEqual(candle_data.first().timestamp, self.timestamp_from)
        self.assert_combined_candle([filtered_1, filtered_2])
