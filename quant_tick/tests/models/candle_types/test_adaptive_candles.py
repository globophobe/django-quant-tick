from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import time_machine
from django.test import TestCase

from quant_tick.constants import FileData, Frequency, SampleType
from quant_tick.lib import get_current_time, get_min_time
from quant_tick.models import (
    AdaptiveCandle,
    CandleCache,
    CandleData,
    TaskState,
    TradeData,
)
from quant_tick.services.aggregate_candles import aggregate_candle_data
from quant_tick.tests.base import BaseWriteTradeDataTest

from .base import BaseHourIteratorTest, BaseThresholdCandleTest


class AdaptiveCandleTest(BaseWriteTradeDataTest, TestCase):
    def test_cache_target_value_is_updated(self):
        now = get_current_time()
        one_day_ago = now - pd.Timedelta("1d")
        yesterday = one_day_ago.date()
        timestamp_from = get_min_time(one_day_ago, "1d")
        timestamp_to = timestamp_from + pd.Timedelta("1d")
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
        raw = self.get_raw(
            timestamp_from,
            price=Decimal(1),
            notional=Decimal(123),
        )
        TradeData.write(
            symbol,
            timestamp_from,
            timestamp_to,
            pd.DataFrame([]),
            raw_trades=raw,
        )
        cache = candle.get_cache_data(now, {"date": yesterday, "target_value": 0})
        self.assertEqual(cache["target_value"], 123)

    def test_cache_target_value_is_not_updated(self):
        candle = AdaptiveCandle(
            json_data={"source_data": FileData.RAW, "cache_reset": Frequency.DAY}
        )
        now = get_current_time()
        cache = candle.get_cache_data(now, {"date": now.date(), "target_value": 123})
        self.assertEqual(cache["target_value"], 123)


@time_machine.travel(datetime(2009, 1, 4, tzinfo=UTC), tick=False)
@patch(
    "quant_tick.models.candles.get_current_time",
    return_value=datetime(2009, 1, 4, 3, tzinfo=UTC),
)
class AdaptiveNotionalCandleTest(
    BaseHourIteratorTest,
    BaseThresholdCandleTest,
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

    def test_one_candle_from_one_trade_in_the_first_hour_then_two_trades_with_retry(
        self, mock_get_current_time
    ):
        filtered = self.get_filtered(self.timestamp_from, notional=Decimal(1))
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

    def test_callback_crosses_gaps_in_trades_and_adaptive_history(
        self, mock_get_current_time
    ):
        history = TradeData.objects.get(symbol=self.symbol)
        history.frequency = Frequency.HOUR
        history.save(update_fields=["frequency"])
        TradeData.objects.bulk_create(
            [
                TradeData(
                    symbol=self.symbol,
                    timestamp=history.timestamp + pd.Timedelta(hours=hour),
                    frequency=Frequency.HOUR,
                    json_data={"candle": {"notional": 0}},
                )
                for hour in range(1, 24)
                if hour != 12
            ]
        )
        self.write_trade_data(
            self.timestamp_from,
            self.one_hour_from_now,
            self.get_filtered(self.timestamp_from, notional=Decimal("0.5")),
        )
        self.write_trade_data(
            self.two_hours_from_now,
            self.three_hours_from_now,
            self.get_filtered(self.two_hours_from_now, notional=Decimal("0.5")),
        )
        self.assertFalse(self.candle.has_moving_average_history(self.timestamp_from))

        with (
            patch(
                "quant_tick.services.aggregate_candles.get_current_time",
                return_value=self.three_hours_from_now,
            ),
            self.assertLogs("quant_tick.models.candles", level="ERROR") as logs,
        ):
            response = aggregate_candle_data([{}])

        self.assertEqual(response, {"ok": True, "processed": 1})
        self.assertEqual(len(logs.records), 2)
        for record in logs.records:
            message = record.getMessage()
            self.assertIn("60.0 missing minutes", message)
            self.assertIn(self.candle.code_name, message)
            self.assertIn(self.symbol.exchange, message)
            self.assertIn(self.symbol.api_symbol, message)
        row = CandleData.objects.get(candle=self.candle)
        self.assertEqual(row.timestamp, self.timestamp_from)
        self.assertEqual(row.notional, Decimal(1))
        self.assertEqual(row.ticks, 2)
        cache = CandleCache.objects.filter(candle=self.candle).last()
        self.assertEqual(cache.timestamp, self.two_hours_from_now)
        self.assertEqual(cache.frequency, Frequency.HOUR)
        self.assertEqual(cache.json_data["target_value"], 1)
        self.assertEqual(cache.json_data["sample_value"], 0)
        state = TaskState.objects.get()
        self.assertEqual(state.recent_error_count, 0)
        self.assertIsNone(state.next_fetch_at)

        with (
            patch(
                "quant_tick.services.aggregate_candles.get_current_time",
                return_value=self.three_hours_from_now + pd.Timedelta("1h"),
            ),
            self.assertNoLogs("quant_tick.models.candles", level="WARNING"),
        ):
            aggregate_candle_data([{}])

        self.assertEqual(CandleData.objects.count(), 1)
        self.assertEqual(CandleCache.objects.count(), 2)
        self.assertEqual(CandleCache.objects.last().pk, cache.pk)

        TradeData.objects.filter(
            symbol=self.symbol,
            timestamp=history.timestamp + pd.Timedelta("13h"),
        ).delete()
        self.assertFalse(
            self.candle.has_moving_average_history(
                self.timestamp_from, max_gap=timedelta(hours=1)
            )
        )

    def test_gap_tolerant_history_keeps_warmup_and_requires_positive_activity(
        self, mock_get_current_time
    ):
        history = TradeData.objects.get(symbol=self.symbol)
        cases = (
            ("partial warmup", pd.Timedelta("23h59min"), Frequency.DAY, Decimal(1)),
            ("empty window", pd.Timedelta("2d"), Frequency.DAY, Decimal(1)),
            ("zero activity", pd.Timedelta("1d"), Frequency.DAY, Decimal(0)),
            ("large history gap", pd.Timedelta("1d"), Frequency.HOUR, Decimal(1)),
        )
        for label, age, frequency, notional in cases:
            with self.subTest(label=label):
                history.timestamp = self.timestamp_from - age
                history.frequency = frequency
                history.json_data = {"candle": {"notional": notional}}
                history.save(update_fields=["timestamp", "frequency", "json_data"])
                self.assertFalse(
                    self.candle.has_moving_average_history(
                        self.timestamp_from, max_gap=timedelta(hours=1)
                    )
                )

    def test_callback_stops_at_large_gap_and_resumes_after_repair(
        self, mock_get_current_time
    ):
        later_from = self.two_hours_from_now + pd.Timedelta("1min")
        later_to = later_from + pd.Timedelta("1min")
        self.write_trade_data(
            self.timestamp_from,
            self.one_hour_from_now,
            self.get_filtered(self.timestamp_from, notional=Decimal("0.5")),
        )
        self.write_trade_data(
            later_from,
            later_to,
            self.get_filtered(later_from, notional=Decimal("0.25")),
        )

        with (
            patch(
                "quant_tick.services.aggregate_candles.get_current_time",
                return_value=self.three_hours_from_now,
            ),
            self.assertLogs("quant_tick.models.candles", level="ERROR") as logs,
        ):
            aggregate_candle_data([{}])

        self.assertEqual(len(logs.records), 1)
        self.assertIn("stopped on oversized TradeData gap", logs.output[0])
        self.assertIn("61.0 missing minutes", logs.output[0])
        self.assertFalse(CandleData.objects.exists())
        cache = CandleCache.objects.get(candle=self.candle)
        self.assertEqual(cache.timestamp, self.timestamp_from)
        self.assertEqual(cache.frequency, Frequency.HOUR)
        self.assertEqual(cache.json_data["sample_value"], Decimal("0.5"))
        self.assertIsNone(TaskState.objects.get().next_fetch_at)

        self.write_trade_data(
            self.one_hour_from_now,
            self.two_hours_from_now,
            self.get_filtered(self.one_hour_from_now, notional=Decimal("0.25")),
        )
        with (
            patch(
                "quant_tick.services.aggregate_candles.get_current_time",
                return_value=self.three_hours_from_now,
            ),
            self.assertLogs("quant_tick.models.candles", level="ERROR") as logs,
        ):
            aggregate_candle_data([{}])

        self.assertEqual(len(logs.records), 1)
        self.assertIn("skipping TradeData gap", logs.output[0])
        self.assertIn("1.0 missing minutes", logs.output[0])
        row = CandleData.objects.get(candle=self.candle)
        self.assertEqual(row.notional, Decimal(1))
        self.assertEqual(row.ticks, 3)
        self.assertEqual(row.timestamp, self.timestamp_from)
        cache = CandleCache.objects.filter(candle=self.candle).last()
        self.assertEqual(cache.timestamp, later_from)
        self.assertEqual(cache.frequency, Frequency.MINUTE)
