from datetime import datetime

import pandas as pd
import time_machine
from django.test import TestCase

from quant_tick.constants import FileData, Frequency, SampleType
from quant_tick.lib import get_current_time, get_min_time, get_previous_time
from quant_tick.models import Candle, CandleCache, TradeData
from quant_tick.storage import convert_candle_cache_to_daily

from ..base import BaseSymbolTest, BaseWriteTradeDataTest
from .candle_types.base import BaseDayIteratorTest


class CandleDataFrameTest(BaseWriteTradeDataTest, TestCase):
    def setUp(self):
        super().setUp()
        self.timestamp_to = self.timestamp_from + pd.Timedelta("1min")
        self.symbol = self.get_symbol("test")
        self.candle = Candle.objects.create(
            symbol=self.symbol, json_data={"source_data": FileData.RAW}
        )

    def test_get_data_frame(self):
        filtered = self.get_filtered(self.timestamp_from)
        TradeData.write(
            self.symbol,
            self.timestamp_from,
            self.timestamp_to,
            filtered,
            pd.DataFrame([]),
        )
        trade_data = TradeData.objects.all()
        self.assertEqual(trade_data.count(), 1)
        t = trade_data[0]
        data_frame = t.get_data_frame(FileData.RAW)
        df = self.candle.get_data_frame(self.timestamp_from, self.timestamp_to, t)
        self.assertTrue(all(data_frame.columns == df.columns))
        self.assertTrue(all(data_frame == df))


@time_machine.travel(datetime(2009, 1, 4), tick=False)
class CandleInitializeTest(BaseSymbolTest, BaseDayIteratorTest, TestCase):
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


class CandleCacheTest(BaseSymbolTest, TestCase):
    def setUp(self):
        super().setUp()
        self.candle = Candle.objects.create(
            symbol=self.get_symbol(), json_data={"sample_type": SampleType.NOTIONAL}
        )

    def test_convert_candle_cache_to_daily(self):
        timestamp_to = get_min_time(get_current_time(), value="1d")
        timestamp_from = get_previous_time(timestamp_to, value="1d")
        total = 24
        target_value = 25
        for value in range(total):
            ts = timestamp_from + pd.Timedelta(f"{value}h")
            val = value + 1
            expected_next = {
                "open": 0,
                "high": val,
                "low": -val,
                "close": 1,
                "volume": val * 1000,
                "buyVolume": val * 500,
                "notional": val * 100,
                "buyNotional": val * 50,
                "ticks": val * 10,
                "buyTicks": val * 5,
            }
            CandleCache.objects.create(
                candle=self.candle,
                timestamp=ts,
                frequency=Frequency.HOUR,
                json_data={
                    "sample_value": val,
                    "target_value": target_value,
                    "next": expected_next,
                },
            )
        convert_candle_cache_to_daily(self.candle)
        candle_cache = CandleCache.objects.filter(candle=self.candle)
        self.assertFalse(candle_cache.filter(frequency=Frequency.HOUR).exists())
        daily = candle_cache.filter(frequency=Frequency.DAY)
        self.assertEqual(daily.count(), 1)
        daily = daily[0]
        self.assertEqual(daily.timestamp, timestamp_from)
        self.assertEqual(daily.json_data["sample_value"], total)
        self.assertEqual(daily.json_data["target_value"], target_value)
        self.assertEqual(daily.json_data["next"], expected_next)

    def test_candle_cache_is_not_converted_to_daily_without_all_timestamps(self):
        timestamp_to = get_min_time(get_current_time(), value="1d")
        timestamp_from = get_previous_time(timestamp_to, value="1d")
        CandleCache.objects.create(
            candle=self.candle,
            timestamp=timestamp_from,
            frequency=Frequency.HOUR,
            json_data={"sample_value": 0},
        )
        convert_candle_cache_to_daily(self.candle)
        candle_cache = CandleCache.objects.filter(candle=self.candle)
        self.assertFalse(candle_cache.filter(frequency=Frequency.DAY).exists())
        self.assertEqual(candle_cache.filter(frequency=Frequency.HOUR).count(), 1)

    def test_convert_candle_cache_to_daily_with_existing_daily_cache(self):
        timestamp_to = get_min_time(get_current_time(), value="1d")
        day_three_from = get_previous_time(timestamp_to, value="3d")
        day_two_from = get_previous_time(timestamp_to, value="2d")
        day_one_from = get_previous_time(timestamp_to, value="1d")
        CandleCache.objects.create(
            candle=self.candle,
            timestamp=timestamp_to,
            frequency=Frequency.DAY,
            json_data={"sample_value": 24},
        )
        for day_from in (day_three_from, day_two_from):
            for hour in range(24):
                CandleCache.objects.create(
                    candle=self.candle,
                    timestamp=day_from + pd.Timedelta(f"{hour}h"),
                    frequency=Frequency.HOUR,
                    json_data={"sample_value": hour + 1},
                )

        convert_candle_cache_to_daily(self.candle)

        candle_cache = CandleCache.objects.filter(candle=self.candle)
        self.assertEqual(candle_cache.filter(frequency=Frequency.DAY).count(), 3)
        self.assertFalse(
            candle_cache.filter(
                frequency=Frequency.HOUR,
                timestamp__gte=day_three_from,
                timestamp__lt=timestamp_to,
            ).exists()
        )
