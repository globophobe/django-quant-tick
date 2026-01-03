import pandas as pd
from django.test import TestCase

from quant_tick.constants import FileData, Frequency, SampleType
from quant_tick.lib import get_current_time
from quant_tick.models import AdaptiveCandle, TradeData
from quant_tick.tests.base import BaseWriteTradeDataTest


class AdaptiveCandleTest(BaseWriteTradeDataTest, TestCase):
    def test_cache_target_value_is_updated(self):
        """If not same day, cache target value is updated."""
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
        filtered = self.get_filtered(one_day_ago, price=1, notional=123)
        trade_data = TradeData.objects.create(
            symbol=symbol, timestamp=one_day_ago, frequency=Frequency.DAY
        )
        TradeData.write_data_frame(trade_data, filtered, pd.DataFrame([]))
        cache = candle.get_cache_data(now, {"date": yesterday, "target_value": 0})
        self.assertEqual(cache["target_value"], 123)

    def test_cache_target_value_is_not_updated(self):
        """If same day, cache target value is not updated."""
        candle = AdaptiveCandle(
            json_data={"source_data": FileData.RAW, "cache_reset": Frequency.DAY}
        )
        now = get_current_time()
        cache = candle.get_cache_data(now, {"date": now.date(), "target_value": 123})
        self.assertEqual(cache["target_value"], 123)
