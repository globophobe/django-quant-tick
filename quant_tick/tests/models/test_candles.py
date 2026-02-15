import pandas as pd
from django.test import TestCase

from quant_tick.constants import Exchange, FileData, Frequency, SampleType
from quant_tick.lib import get_current_time, get_min_time, get_previous_time
from quant_tick.models import Candle, CandleCache, Symbol, TradeData
from quant_tick.storage import convert_candle_cache_to_daily

from ..base import BaseSymbolTest, BaseWriteTradeDataTest


class BaseCandleTest(TestCase):
    def get_symbol(self, name: str, exchange: Exchange = Exchange.COINBASE) -> Symbol:
        """Get symbol."""
        return Symbol.objects.create(
            global_symbol=self.global_symbol,
            exchange=exchange,
            api_symbol=name,
        )


class CandleTest(BaseWriteTradeDataTest, BaseCandleTest):
    def setUp(self):
        super().setUp()
        self.timestamp_to = self.timestamp_from + pd.Timedelta("1min")
        self.symbol = self.get_symbol("test")
        self.candle = Candle.objects.create(
            symbol=self.symbol, json_data={"source_data": FileData.RAW}
        )

    def test_get_data_frame(self):
        """Get data frame."""
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


class CandleCacheTest(BaseSymbolTest, BaseCandleTest):
    def setUp(self):
        super().setUp()
        self.symbol = self.get_symbol("test")
        self.candle = Candle.objects.create(
            symbol=self.symbol, json_data={"sample_type": SampleType.NOTIONAL}
        )

    def test_convert_candle_cache_to_daily(self):
        """Convert candle cache to daily."""
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
        """Candle cache is not converted to daily, without all timestamps."""
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
