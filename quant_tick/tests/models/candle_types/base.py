from datetime import datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.constants import FileData, Frequency
from quant_tick.lib import get_current_time, get_min_time
from quant_tick.models import Candle, CandleCache, CandleData, TradeData
from quant_tick.tests.base import BaseSymbolTest, BaseWriteTradeDataTest


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


class BaseTradeDataCandleTest(BaseWriteTradeDataTest, BaseCandleCacheIteratorTest):
    def write_trade_data(
        self, timestamp_from: datetime, timestamp_to: datetime, data_frame: DataFrame
    ) -> None:
        TradeData.write(
            self.symbol,
            timestamp_from,
            timestamp_to,
            pd.DataFrame([]),
            raw_trades=data_frame,
        )


class BaseThresholdCandleTest(BaseTradeDataCandleTest):
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
        filtered = self.get_filtered(self.timestamp_from, notional=Decimal(1))
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered)
        self.candle.candles(self.timestamp_from, self.one_hour_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)

    def test_one_candle_from_trade_in_the_first_hour_with_retry(
        self, mock_get_current_time
    ):
        filtered = self.get_filtered(self.timestamp_from, notional=Decimal(1))
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
        filtered_1 = self.get_filtered(self.timestamp_from, notional=Decimal(1))
        self.write_trade_data(self.timestamp_from, self.one_hour_from_now, filtered_1)
        filtered_2 = self.get_filtered(self.one_hour_from_now, notional=Decimal(1))
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
            self.get_filtered(self.timestamp_from, notional=Decimal(1)),
        )
        TradeData.objects.create(
            symbol=self.symbol,
            timestamp=self.one_hour_from_now,
            frequency=Frequency.HOUR,
        )
        self.write_trade_data(
            self.two_hours_from_now,
            self.three_hours_from_now,
            self.get_filtered(self.two_hours_from_now, notional=Decimal(1)),
        )
        self.candle.candles(self.timestamp_from, self.three_hours_from_now)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 2)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
        self.assertEqual(candle_data[1].timestamp, self.two_hours_from_now)
