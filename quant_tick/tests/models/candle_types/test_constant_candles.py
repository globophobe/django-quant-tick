from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import time_machine
from django.test import TestCase

from quant_tick.constants import FileData, Frequency, SampleType
from quant_tick.lib import get_current_time
from quant_tick.models import CandleCache, CandleData, ConstantCandle, TradeData

from .base import BaseDayIteratorTest, BaseHourIteratorTest, BaseTradeDataCandleTest


class ConstantCandleTest(TestCase):
    def test_daily_cache_reset(self):
        now = get_current_time()
        one_day_ago = now - pd.Timedelta("1d")
        candle = ConstantCandle(
            json_data={"source_data": FileData.RAW, "cache_reset": Frequency.DAY}
        )
        cache = candle.get_cache_data(
            now, {"date": one_day_ago.date(), "sample_value": 123}
        )
        self.assertEqual(cache["sample_value"], 0)

    def test_daily_cache_does_not_reset(self):
        now = get_current_time()
        candle = ConstantCandle(
            json_data={"source_data": FileData.RAW, "cache_reset": Frequency.DAY}
        )
        cache = candle.get_cache_data(now, {"date": now.date(), "sample_value": 123})
        self.assertEqual(cache["sample_value"], 123)

    def test_weekly_cache_reset(self):
        now = get_current_time()
        days = 7 - now.date().weekday() % 7
        next_monday = now + pd.Timedelta(f"{days}d")
        candle = ConstantCandle(
            json_data={"source_data": FileData.RAW, "cache_reset": Frequency.WEEK}
        )
        cache = candle.get_cache_data(
            next_monday, {"date": now.date(), "sample_value": 123}
        )
        self.assertEqual(cache["sample_value"], 0)

    def test_weekly_cache_does_not_reset(self):
        now = get_current_time()
        days = 6 - now.date().weekday() % 7
        next_sunday = now + pd.Timedelta(f"{days}d")
        candle = ConstantCandle(json_data={"cache_reset": Frequency.WEEK})
        cache = candle.get_cache_data(
            next_sunday, {"date": next_sunday.date(), "sample_value": 123}
        )
        self.assertEqual(cache["sample_value"], 123)


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.models.candles.get_current_time",
    return_value=datetime(2009, 1, 4, 3).replace(tzinfo=UTC),
)
class ConstantNotionalHourFrequencyCandleTest(
    BaseHourIteratorTest,
    BaseTradeDataCandleTest,
    TestCase,
):
    def get_candle(self) -> ConstantCandle:
        return ConstantCandle.objects.create(
            symbol=self.symbol,
            json_data={
                "source_data": FileData.RAW,
                "sample_type": SampleType.NOTIONAL,
                "target_value": 1,
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
        querysets = {
            TradeData: TradeData.objects.all(),
            CandleCache: CandleCache.objects.all(),
            CandleData: CandleData.objects.all(),
        }
        for model, queryset in querysets.items():
            self.assertEqual(queryset.count(), 1)
        self.assertEqual(querysets[CandleData][0].timestamp, self.timestamp_from)

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

    def test_no_candles_without_prior_cache(self, mock_get_current_time):
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
        self.assertEqual(CandleCache.objects.count(), 1)


@time_machine.travel(datetime(2009, 1, 4), tick=False)
@patch(
    "quant_tick.models.candles.get_current_time",
    return_value=datetime(2009, 1, 5).replace(tzinfo=UTC),
)
class ConstantNotionalDayFrequencyIrregularCandleTest(
    BaseDayIteratorTest,
    BaseTradeDataCandleTest,
    TestCase,
):
    def get_candle(self) -> ConstantCandle:
        return ConstantCandle.objects.create(
            symbol=self.symbol,
            json_data={
                "source_data": FileData.RAW,
                "sample_type": SampleType.NOTIONAL,
                "target_value": 1,
                "cache_reset": Frequency.DAY,
            },
        )

    def test_one_incomplete_candle(self, mock_get_current_time):
        last_hour = self.timestamp_from + pd.Timedelta("23h")
        filtered = self.get_filtered(last_hour, notional=Decimal("0.5"))
        self.write_trade_data(last_hour, self.one_day_from_now, filtered)
        self.candle.candles(last_hour, self.one_day_from_now)
        self.assertEqual(CandleCache.objects.count(), 1)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, last_hour)
        self.assertTrue(candle_data[0].incomplete)
