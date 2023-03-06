import pandas as pd
from django.test import TestCase

from quant_candles.constants import Frequency
from quant_candles.lib import get_current_time
from quant_candles.models import ConstantCandle


class ConstantCandleTest(TestCase):
    databases = {"default", "read_only"}

    def test_daily_cache_reset(self):
        """If not same day, daily cache resets."""
        now = get_current_time()
        one_day_ago = now - pd.Timedelta("1d")
        candle = ConstantCandle(json_data={"cache_reset": Frequency.DAY})
        cache = candle.get_cache_data(
            now, {"date": one_day_ago.date(), "sample_value": 123}
        )
        self.assertEqual(cache["sample_value"], 0)

    def test_daily_cache_does_not_reset(self):
        """If same day, daily cache does not reset."""
        now = get_current_time()
        candle = ConstantCandle(json_data={"cache_reset": Frequency.DAY})
        cache = candle.get_cache_data(now, {"date": now.date(), "sample_value": 123})
        self.assertEqual(cache["sample_value"], 123)

    def test_weekly_cache_reset(self):
        """If not same week, weekly cache resets."""
        now = get_current_time()
        days = 7 - now.date().weekday() % 7
        next_monday = now + pd.Timedelta(f"{days}d")
        candle = ConstantCandle(json_data={"cache_reset": Frequency.WEEK})
        cache = candle.get_cache_data(
            next_monday, {"date": now.date(), "sample_value": 123}
        )
        self.assertEqual(cache["sample_value"], 0)

    def test_weekly_cache_does_not_reset(self):
        """If same week, weekly cache does not reset."""
        now = get_current_time()
        days = 6 - now.date().weekday() % 7
        next_sunday = now + pd.Timedelta(f"{days}d")
        candle = ConstantCandle(json_data={"cache_reset": Frequency.WEEK})
        cache = candle.get_cache_data(
            next_sunday, {"date": next_sunday.date(), "sample_value": 123}
        )
        self.assertEqual(cache["sample_value"], 123)
