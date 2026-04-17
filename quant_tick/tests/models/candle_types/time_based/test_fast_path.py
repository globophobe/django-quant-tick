from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

import pandas as pd
from django.test import TestCase

from quant_tick.constants import FileData, Frequency
from quant_tick.lib import (
    aggregate_candle,
    get_current_time,
    get_min_time,
    is_decimal_close,
    merge_cache,
)
from quant_tick.models import CandleData, TimeBasedCandle, TradeData
from quant_tick.tests.base import BaseWriteTradeDataTest


class TimeBasedCandleFastPathTest(BaseWriteTradeDataTest, TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.day_from = get_min_time(get_current_time(), "1d") - pd.Timedelta("2d")
        self.one_day_from = self.day_from + pd.Timedelta("1d")
        self.two_days_from = self.day_from + pd.Timedelta("2d")

    def test_get_trade_candle_uses_aligned_daily_window(self):
        symbol = self.get_symbol(save_raw=False)
        raw = self.get_raw(
            self.day_from,
            price=Decimal("1000"),
            notional=Decimal("1"),
            tick_rule=1,
        )
        payload = aggregate_candle(
            raw,
            min_volume_exponent=1,
            min_notional_exponent=1,
        )
        trade_data = TradeData.objects.create(
            symbol=symbol,
            timestamp=self.day_from,
            frequency=Frequency.DAY,
            json_data={"candle": payload},
        )
        candle = TimeBasedCandle.objects.create(
            symbol=symbol,
            json_data={
                "source_data": FileData.RAW,
                "window": "1d",
                "min_volume_exponent": 1,
                "min_notional_exponent": 1,
            },
        )

        with (
            patch.object(
                TradeData,
                "get_candle_source_data",
                return_value=FileData.RAW,
            ),
            patch.object(
                TradeData,
                "get_data_frame",
                side_effect=AssertionError("should not read parquet"),
            ),
        ):
            data = candle.get_trade_candle(
                self.day_from, self.one_day_from, trade_data
            )

        self.assertEqual(data, payload)

    def test_get_trade_candle_returns_none_when_source_data_differs(self):
        symbol = self.get_symbol(save_raw=False)
        trade_data = TradeData.objects.create(
            symbol=symbol,
            timestamp=self.day_from,
            frequency=Frequency.DAY,
            json_data={
                "candle": aggregate_candle(
                    self.get_raw(
                        self.day_from,
                        price=Decimal("1000"),
                        notional=Decimal("1"),
                        tick_rule=1,
                    ),
                    min_volume_exponent=1,
                    min_notional_exponent=1,
                )
            },
        )
        candle = TimeBasedCandle.objects.create(
            symbol=symbol,
            json_data={
                "source_data": FileData.RAW,
                "window": "1d",
                "min_volume_exponent": 1,
                "min_notional_exponent": 1,
            },
        )
        with (
            patch.object(
                TradeData,
                "get_candle_source_data",
                return_value=FileData.FILTERED,
            ),
        ):
            data = candle.get_trade_candle(
                self.day_from, self.one_day_from, trade_data
            )

        self.assertIsNone(data)

    def test_get_trade_candle_returns_none_when_thresholds_do_not_match(self):
        symbol = self.get_symbol(save_raw=False)
        trade_data = TradeData.objects.create(
            symbol=symbol,
            timestamp=self.day_from,
            frequency=Frequency.DAY,
            json_data={
                "candle": aggregate_candle(
                    self.get_raw(
                        self.day_from,
                        price=Decimal("1000"),
                        notional=Decimal("1"),
                        tick_rule=1,
                    ),
                    min_volume_exponent=1,
                    min_notional_exponent=1,
                )
            },
        )
        candle = TimeBasedCandle.objects.create(
            symbol=symbol,
            json_data={
                "source_data": FileData.RAW,
                "window": "1d",
                "min_volume_exponent": 2,
                "min_notional_exponent": 1,
            },
        )
        with (
            patch.object(
                TradeData,
                "get_candle_source_data",
                return_value=FileData.RAW,
            ),
        ):
            data = candle.get_trade_candle(
                self.day_from, self.one_day_from, trade_data
            )

        self.assertIsNone(data)

    def test_candles_merge_trade_candles_for_multi_day_window(self):
        symbol = self.get_symbol(save_raw=False)
        first_raw = self.get_raw(
            self.day_from,
            price=Decimal("1000"),
            notional=Decimal("1"),
            tick_rule=1,
        )
        second_raw = self.get_raw(
            self.one_day_from,
            price=Decimal("2000"),
            notional=Decimal("1"),
            tick_rule=1,
        )
        first_payload = aggregate_candle(
            first_raw,
            min_volume_exponent=1,
            min_notional_exponent=1,
        )
        second_payload = aggregate_candle(
            second_raw,
            min_volume_exponent=1,
            min_notional_exponent=1,
        )
        TradeData.objects.create(
            symbol=symbol,
            timestamp=self.day_from,
            frequency=Frequency.DAY,
            json_data={"candle": first_payload},
        )
        TradeData.objects.create(
            symbol=symbol,
            timestamp=self.one_day_from,
            frequency=Frequency.DAY,
            json_data={"candle": second_payload},
        )
        expected = merge_cache(
            dict(first_payload),
            dict(second_payload),
        )
        candle = TimeBasedCandle.objects.create(
            symbol=symbol,
            json_data={
                "source_data": FileData.RAW,
                "window": "2d",
                "min_volume_exponent": 1,
                "min_notional_exponent": 1,
            },
        )

        with (
            patch.object(
                TradeData,
                "get_candle_source_data",
                return_value=FileData.RAW,
            ),
            patch.object(
                TradeData,
                "get_data_frame",
                side_effect=AssertionError("should not read parquet"),
            ),
        ):
            candle.candles(self.day_from, self.two_days_from)

        stored = CandleData.objects.get(candle=candle)
        self.assertEqual(stored.timestamp, self.day_from)
        self.assertEqual(stored.open, expected["open"])
        self.assertEqual(stored.high, expected["high"])
        self.assertEqual(stored.low, expected["low"])
        self.assertEqual(stored.close, expected["close"])
        self.assertEqual(stored.volume, expected["volume"])
        self.assertEqual(stored.buy_volume, expected["buyVolume"])
        self.assertEqual(stored.notional, expected["notional"])
        self.assertEqual(stored.buy_notional, expected["buyNotional"])
        self.assertEqual(stored.ticks, expected["ticks"])
        self.assertEqual(stored.buy_ticks, expected["buyTicks"])
        self.assertTrue(
            is_decimal_close(stored.realized_variance, expected["realizedVariance"])
        )
        self.assertEqual(
            stored.extra_data,
            {
                "roundVolume": expected["roundVolume"],
                "roundBuyVolume": expected["roundBuyVolume"],
                "roundVolumeSumNotional": expected["roundVolumeSumNotional"],
                "roundBuyVolumeSumNotional": expected["roundBuyVolumeSumNotional"],
                "roundNotional": expected["roundNotional"],
                "roundBuyNotional": expected["roundBuyNotional"],
                "roundNotionalSumVolume": expected["roundNotionalSumVolume"],
                "roundBuyNotionalSumVolume": expected["roundBuyNotionalSumVolume"],
            },
        )
