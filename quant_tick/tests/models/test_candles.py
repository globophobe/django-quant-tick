from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import time_machine
from django.test import TestCase

from quant_tick.constants import FileData, Frequency, SampleType
from quant_tick.models import Candle, CandleCache, CandleData, ConstantCandle, TradeData
from quant_tick.services.task_lease import TaskLeaseLost

from ..base import BaseSymbolTest, BaseWriteTradeDataTest
from .candle_types.base import (
    BaseDayIteratorTest,
    BaseHourIteratorTest,
    BaseTradeDataCandleTest,
)


class CandleDataFrameTest(BaseWriteTradeDataTest, TestCase):
    def setUp(self):
        super().setUp()
        self.timestamp_to = self.timestamp_from + pd.Timedelta("1min")
        self.symbol = self.get_symbol("test")
        self.candle = Candle.objects.create(
            symbol=self.symbol, json_data={"source_data": FileData.RAW}
        )

    def test_get_data_frame(self):
        raw = self.get_raw(self.timestamp_from)
        TradeData.write(
            self.symbol,
            self.timestamp_from,
            self.timestamp_to,
            pd.DataFrame([]),
            raw_trades=raw,
        )
        trade_data = TradeData.objects.all()
        self.assertEqual(trade_data.count(), 1)
        t = trade_data[0]
        data_frame = t.get_data_frame(FileData.RAW)
        df = self.candle.get_data_frame(self.timestamp_from, self.timestamp_to, t)
        self.assertTrue(all(data_frame.columns == df.columns))
        self.assertTrue(all(data_frame == df))


@time_machine.travel(datetime(2009, 1, 4, tzinfo=UTC), tick=False)
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

    def test_initial_timestamp_to_is_clamped_by_candle_date_to(self):
        self.candle.date_to = self.two_days_from_now.date()
        cases = (
            (self.one_day_from_now, self.one_day_from_now),
            (self.two_days_from_now, self.two_days_from_now),
            (self.three_days_from_now, self.two_days_from_now),
        )

        for requested_to, expected_to in cases:
            with self.subTest(requested_to=requested_to):
                _timestamp_from, timestamp_to, _data = self.candle.initialize(
                    self.timestamp_from,
                    requested_to,
                )
                self.assertEqual(timestamp_to, expected_to)

    def test_initial_timestamp_from_with_candle_date_from(self):
        self.candle.date_from = self.one_day_from_now.date()
        timestamp_from, _timestamp_to, _ = self.candle.initialize(
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

    def test_retry_initial_timestamp_uses_preserved_cache_boundary(self):
        self.create_candle_cache(self.timestamp_from)
        retry_from = self.one_day_from_now + pd.Timedelta("12h")

        timestamp_from, timestamp_to, _ = self.candle.initialize(
            retry_from, self.three_days_from_now, retry=True
        )

        self.assertEqual(timestamp_from, self.one_day_from_now)
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


@time_machine.travel(datetime(2009, 1, 4, tzinfo=UTC), tick=False)
@patch(
    "quant_tick.models.candles.get_current_time",
    return_value=datetime(2009, 1, 4, 3, tzinfo=UTC),
)
class CandleTransactionTest(BaseHourIteratorTest, BaseTradeDataCandleTest, TestCase):
    def get_candle(self) -> ConstantCandle:
        return ConstantCandle.objects.create(
            symbol=self.symbol,
            json_data={
                "source_data": FileData.RAW,
                "sample_type": SampleType.NOTIONAL,
                "target_value": 1,
            },
        )

    def test_cache_is_rolled_back_when_data_write_fails(self, mock_get_current_time):
        filtered = self.get_filtered(self.timestamp_from, notional=Decimal(1))
        self.write_trade_data(
            self.timestamp_from,
            self.one_hour_from_now,
            filtered,
        )

        with (
            patch.object(
                self.candle,
                "write_data",
                side_effect=RuntimeError("boom"),
            ),
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            self.candle.candles(self.timestamp_from, self.one_hour_from_now)

        self.assertFalse(CandleCache.objects.exists())
        self.assertFalse(CandleData.objects.exists())

    def test_lease_loss_stops_before_next_partition_commit(self, mock_get_current_time):
        first = self.get_filtered(self.timestamp_from, notional=Decimal(1))
        second = self.get_filtered(self.one_hour_from_now, notional=Decimal(1))
        self.write_trade_data(
            self.timestamp_from,
            self.one_hour_from_now,
            first,
        )
        self.write_trade_data(
            self.one_hour_from_now,
            self.two_hours_from_now,
            second,
        )
        assertion_count = 0

        def assert_lease_owned():
            nonlocal assertion_count
            assertion_count += 1
            if assertion_count == 2:
                raise TaskLeaseLost("lease ownership lost")

        with self.assertRaisesRegex(TaskLeaseLost, "ownership lost"):
            self.candle.candles(
                self.timestamp_from,
                self.two_hours_from_now,
                assert_lease_owned=assert_lease_owned,
            )

        self.assertEqual(CandleCache.objects.count(), 1)
        candle_data = CandleData.objects.all()
        self.assertEqual(candle_data.count(), 1)
        self.assertEqual(candle_data[0].timestamp, self.timestamp_from)
