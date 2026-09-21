from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import time_machine
from django.test import TestCase

from quant_tick.constants import FileData, Frequency, SampleType
from quant_tick.models import (
    AdaptiveCandle,
    CandleCache,
    CandleData,
    ConstantCandle,
    TimeBasedCandle,
    TradeData,
)
from quant_tick.services.aggregate_candles import aggregate_candle_data
from quant_tick.tests.base import BaseWriteTradeDataTest


@time_machine.travel(datetime(2026, 5, 2, tzinfo=UTC), tick=False)
class CandleGapBoundaryTest(BaseWriteTradeDataTest, TestCase):
    def write_partition(self, symbol, timestamp, frequency, notional, price):
        TradeData.write(
            symbol,
            timestamp,
            timestamp + timedelta(minutes=frequency),
            pd.DataFrame([]),
            raw_trades=self.get_raw(
                timestamp,
                notional=Decimal(notional),
                price=Decimal(price),
                tick_rule=1,
            ),
        )

    def write_gapped_day(self, symbol, timestamp):
        for hour in range(23):
            self.write_partition(
                symbol, timestamp + timedelta(hours=hour), Frequency.HOUR, 1, 100 + hour
            )

    def test_daily_summary_and_parquet_paths_cross_midnight_without_reusing_trades(
        self,
    ):
        symbol = self.get_symbol()
        day_from = datetime(2026, 1, 3, tzinfo=UTC)
        next_day = day_from + timedelta(days=1)
        until = next_day + timedelta(days=1)
        self.write_gapped_day(symbol, day_from)
        self.write_partition(symbol, next_day, Frequency.DAY, 10, 500)
        candles = [
            TimeBasedCandle.objects.create(
                symbol=symbol,
                json_data={
                    "source_data": FileData.RAW,
                    "window": "1d",
                    "min_volume_exponent": exponent,
                    "min_notional_exponent": exponent,
                },
            )
            for exponent in (1, 2)
        ]
        next_partition = TradeData.objects.get(symbol=symbol, timestamp=next_day)
        self.assertIsNotNone(
            candles[0].get_trade_candle(next_day, until, next_partition)
        )
        self.assertIsNone(candles[1].get_trade_candle(next_day, until, next_partition))
        payload = {"exchange": symbol.exchange, "api_symbol": symbol.api_symbol}

        with patch(
            "quant_tick.services.aggregate_candles.get_current_time",
            return_value=day_from + timedelta(hours=23),
        ):
            aggregate_candle_data([payload])
        self.assertFalse(CandleData.objects.exists())

        for retry in (False, True):
            with self.subTest(retry=retry):
                if retry:
                    payload["timestamp_from"] = next_day
                with (
                    patch(
                        "quant_tick.services.aggregate_candles.get_current_time",
                        return_value=until,
                    ),
                    patch("quant_tick.models.candles.logger"),
                ):
                    aggregate_candle_data([payload])
                for candle in candles:
                    rows = list(
                        CandleData.objects.filter(candle=candle).order_by("timestamp")
                    )
                    self.assertEqual(
                        [row.timestamp for row in rows], [day_from, next_day]
                    )
                    self.assertEqual(
                        [row.notional for row in rows], [Decimal(23), Decimal(10)]
                    )
                    self.assertEqual([row.ticks for row in rows], [23, 1])
                    self.assertEqual(
                        [row.open for row in rows], [Decimal(100), Decimal(500)]
                    )
                    self.assertEqual(
                        [row.close for row in rows], [Decimal(122), Decimal(500)]
                    )
                    self.assertEqual(
                        [row.volume for row in rows], [Decimal(2553), Decimal(5000)]
                    )

    def test_calendar_reset_preserves_partial_constant_and_adaptive_buckets(self):
        boundaries = (
            ("day", datetime(2026, 1, 6, tzinfo=UTC)),
            ("week", datetime(2026, 1, 5, tzinfo=UTC)),
            ("month", datetime(2026, 2, 1, tzinfo=UTC)),
            ("quarter", datetime(2026, 4, 1, tzinfo=UTC)),
            ("year", datetime(2026, 1, 1, tzinfo=UTC)),
        )
        for reset, boundary in boundaries:
            with self.subTest(reset=reset):
                symbol = self.get_symbol(api_symbol=reset)
                day_from = boundary - timedelta(days=1)
                self.write_partition(
                    symbol, day_from - timedelta(days=1), Frequency.DAY, 100, 99
                )
                self.write_gapped_day(symbol, day_from)
                self.write_partition(symbol, boundary, Frequency.HOUR, 10, 500)
                candles = [
                    model.objects.create(
                        symbol=symbol,
                        date_from=day_from.date(),
                        json_data={
                            "source_data": FileData.RAW,
                            "sample_type": SampleType.NOTIONAL,
                            "target_value": 100,
                            "cache_reset": reset,
                            "moving_average_number_of_days": 1,
                            "target_candles_per_day": 1,
                        },
                    )
                    for model in (ConstantCandle, AdaptiveCandle)
                ]
                payload = {"exchange": symbol.exchange, "api_symbol": symbol.api_symbol}
                for retry in (False, True):
                    if retry:
                        payload["timestamp_from"] = boundary
                    with (
                        patch(
                            "quant_tick.services.aggregate_candles.get_current_time",
                            return_value=boundary + timedelta(hours=1),
                        ),
                        patch("quant_tick.models.candles.logger"),
                    ):
                        aggregate_candle_data([payload])
                    for candle in candles:
                        row = CandleData.objects.get(candle=candle)
                        self.assertEqual(row.timestamp, day_from)
                        self.assertTrue(row.incomplete)
                        self.assertEqual(row.notional, Decimal(23))
                        self.assertEqual(row.ticks, 23)
                        self.assertEqual(row.close, Decimal(122))
                        cache = CandleCache.objects.filter(candle=candle).last()
                        self.assertEqual(cache.timestamp, boundary)
                        self.assertEqual(cache.json_data["date"], boundary.date())
                        self.assertEqual(cache.json_data["sample_value"], Decimal(10))
                        self.assertEqual(cache.json_data["next"]["timestamp"], boundary)
                        self.assertEqual(
                            cache.json_data["next"]["notional"], Decimal(10)
                        )
