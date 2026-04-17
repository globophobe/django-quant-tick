from datetime import datetime

import pandas as pd
from pandas import DataFrame

from quant_tick.constants import FileData
from quant_tick.lib import get_current_time, get_min_time
from quant_tick.models import Candle, TradeData
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
            self.symbol, timestamp_from, timestamp_to, data_frame, pd.DataFrame([])
        )
