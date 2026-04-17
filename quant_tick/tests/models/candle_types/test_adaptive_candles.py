from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import time_machine
from django.test import TestCase

from quant_tick.constants import FileData, Frequency, SampleType
from quant_tick.lib import get_current_time, get_min_time
from quant_tick.models import AdaptiveCandle, CandleCache, CandleData, TradeData
from quant_tick.tests.base import BaseWriteTradeDataTest

from .base import BaseHourIteratorTest, BaseTradeDataCandleTest


class AdaptiveCandleTest(BaseWriteTradeDataTest, TestCase):
    def test_cache_target_value_is_updated(self):
        now = get_current_time()
        one_day_ago = now - pd.Timedelta("1d")
        yesterday = one_day_ago.date()
        symbol = self.get_symbol()
        candle = AdaptiveCandle.objects.create(
            symbol=symbol,
            json_data={
                "source_data": FileData.RAW,
                "sample_type": SampleType.VOLUME,
                "moving_average_number_of_days": 1,
                "target_candles_per_day": 1,
                "cache_reset": Frequency.DAY,
            },
        )
        filtered = self.get_filtered(
            one_day_ago,
            price=Decimal("1"),
            notional=Decimal("123"),
        )
        trade_data = TradeData.objects.create(
            symbol=symbol, timestamp=one_day_ago, frequency=Frequency.DAY
        )
        TradeData.write_data_frame(trade_data, filtered, pd.DataFrame([]))
        cache = candle.get_cache_data(now, {"date": yesterday, "target_value": 0})
        self.assertEqual(cache["target_value"], 123)

    def test_cache_target_value_is_not_updated(self):
        candle = AdaptiveCandle(
            json_data={"source_data": FileData.RAW, "cache_reset": Frequency.DAY}
        )
        now = get_current_time()
        cache = candle.get_cache_data(now, {"date": now.date(), "target_value": 123})
        self.assertEqual(cache["target_value"], 123)


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.models.candles.get_current_time",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=UTC),
)
class AdaptiveNotionalCandleTest(
    BaseHourIteratorTest,
    BaseTradeDataCandleTest,
    TestCase,
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

    def get_candle(self) -> AdaptiveCandle:
        return AdaptiveCandle.objects.create(
            symbol=self.symbol,
            json_data={
                "source_data": FileData.RAW,
                "sample_type": SampleType.NOTIONAL,
                "moving_average_number_of_days": 1,
                "target_candles_per_day": 1,
            },
        )

    def test_no_candles_from_trade_in_the_first_hour(self, mock_get_current_time):
        expected = Decimal("0.5")
        filtered = self.get_filtered(self.timestamp_from, notional=expected)
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        self.candle.candles(self.timestamp_from, self.one_hour_from_now)
        self.assertFalse(CandleData.objects.exists())
        candle_cache = CandleCache.objects.all()
        self.assertEqual(candle_cache.count(), 1)
        self.assertEqual(candle_cache[0].json_data["sample_value"], expected)

    def test_one_candle_from_trade_in_the_first_hour(self, mock_get_current_time):
        filtered = self.get_filtered(self.timestamp_from, notional=Decimal("1"))
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        self.candle.candles(self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_one_candle_from_trade_in_the_first_hour_with_retry(
        self, mock_get_current_time
    ):
        filtered = self.get_filtered(self.timestamp_from, notional=Decimal("1"))
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

    def test_one_candle_from_one_trade_in_the_first_hour_then_two_trades_with_retry(
        self, mock_get_current_time
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
        self, mock_get_current_time
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
        self, mock_get_current_time
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
        self, mock_get_current_time
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
