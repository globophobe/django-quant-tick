import pandas as pd
from django.test import TestCase, override_settings

from quant_candles.constants import Frequency
from quant_candles.lib import get_current_time, get_min_time
from quant_candles.models import CandleData, CandleReadOnlyData, ConstantCandle


class ConstantCandleTest(TestCase):
    databases = {"default", "read_only"}

    def test_daily_cache_reset(self):
        """If not same day, daily cache resets."""
        now = get_current_time()
        one_day_ago = now - pd.Timedelta("1d")
        candle = ConstantCandle(json_data={"cache_reset": Frequency.DAY.value})
        cache = candle.get_cache_data(
            now, {"date": one_day_ago.date(), "sample_value": 123}
        )
        self.assertEqual(cache["sample_value"], 0)

    def test_daily_cache_reset_with_incomplete_candle(self):
        """If not same day, daily cache resets, and incomplete candle data saved."""
        timestamp_from = get_min_time(get_current_time(), value="1d")
        one_day_ago = timestamp_from - pd.Timedelta("1d")
        candle = ConstantCandle.objects.create(
            json_data={"cache_reset": Frequency.DAY.value}
        )
        cache = candle.get_cache_data(
            timestamp_from,
            {"date": one_day_ago.date(), "sample_value": 123, "next": {"test": True}},
        )
        self.assertEqual(cache["sample_value"], 0)
        candle_data = CandleReadOnlyData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, timestamp_from - pd.Timedelta("1us"))
        self.assertTrue(candle_data[0].json_data["incomplete"])

    @override_settings(IS_LOCAL=False)
    def test_daily_cache_reset_with_incomplete_candle_not_read_only(self):
        """
        If not same day, daily cache resets,
        and incomplete not read only candle data saved.
        """
        timestamp_from = get_min_time(get_current_time(), value="1d")
        one_day_ago = timestamp_from - pd.Timedelta("1d")
        candle = ConstantCandle.objects.create(
            json_data={"cache_reset": Frequency.DAY.value}
        )
        cache = candle.get_cache_data(
            timestamp_from,
            {"date": one_day_ago.date(), "sample_value": 123, "next": {"test": True}},
        )
        self.assertEqual(cache["sample_value"], 0)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, timestamp_from - pd.Timedelta("1us"))
        self.assertTrue(candle_data[0].json_data["incomplete"])

    def test_daily_cache_does_not_reset(self):
        """If same day, daily cache does not reset."""
        now = get_current_time()
        candle = ConstantCandle(json_data={"cache_reset": Frequency.DAY.value})
        cache = candle.get_cache_data(now, {"date": now.date(), "sample_value": 123})
        self.assertEqual(cache["sample_value"], 123)

    def test_weekly_cache_reset(self):
        """If not same week, weekly cache resets."""
        now = get_current_time()
        days = 7 - now.date().weekday() % 7
        next_monday = now + pd.Timedelta(f"{days}d")
        candle = ConstantCandle(json_data={"cache_reset": Frequency.WEEK.value})
        cache = candle.get_cache_data(
            next_monday, {"date": now.date(), "sample_value": 123}
        )
        self.assertEqual(cache["sample_value"], 0)

    def test_weekly_cache_reset_with_incomplete_candle(self):
        """If not same week, weekly cache resets, and incomplete candle data saved."""
        timestamp_from = get_min_time(get_current_time(), value="1d")
        days = 7 - timestamp_from.date().weekday() % 7
        next_monday = timestamp_from + pd.Timedelta(f"{days}d")
        candle = ConstantCandle.objects.create(
            json_data={"cache_reset": Frequency.WEEK.value}
        )
        cache = candle.get_cache_data(
            next_monday,
            {
                "date": timestamp_from.date(),
                "sample_value": 123,
                "next": {"test": True},
            },
        )
        self.assertEqual(cache["sample_value"], 0)
        candle_data = CandleReadOnlyData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, next_monday - pd.Timedelta("1us"))
        self.assertTrue(candle_data[0].json_data["incomplete"])

    @override_settings(IS_LOCAL=False)
    def test_weekly_cache_reset_with_incomplete_candle_not_read_only(self):
        """
        If not same week, weekly cache resets, and incomplete not read only candle saved.
        """
        timestamp_from = get_min_time(get_current_time(), value="1d")
        days = 7 - timestamp_from.date().weekday() % 7
        next_monday = timestamp_from + pd.Timedelta(f"{days}d")
        candle = ConstantCandle.objects.create(
            json_data={"cache_reset": Frequency.WEEK.value}
        )
        cache = candle.get_cache_data(
            next_monday,
            {
                "date": timestamp_from.date(),
                "sample_value": 123,
                "next": {"test": True},
            },
        )
        self.assertEqual(cache["sample_value"], 0)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, next_monday - pd.Timedelta("1us"))
        self.assertTrue(candle_data[0].json_data["incomplete"])

    def test_weekly_cache_does_not_reset(self):
        """If same week, weekly cache does not reset."""
        now = get_current_time()
        days = 6 - now.date().weekday() % 7
        next_sunday = now + pd.Timedelta(f"{days}d")
        candle = ConstantCandle(json_data={"cache_reset": Frequency.WEEK.value})
        cache = candle.get_cache_data(
            next_sunday, {"date": next_sunday.date(), "sample_value": 123}
        )
        self.assertEqual(cache["sample_value"], 123)
