import json
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.lib import get_next_cache, merge_cache, volume_filter_with_time_window
from quant_tick.lib.candles import aggregate_candle, resample_candles
from quant_tick.models.base import QuantTickDecoder, QuantTickEncoder


class CandleTest(SimpleTestCase):
    def get_data_frame(
        self,
        prices: list[float],
        volumes: list[float] | None = None,
        tick_rules: list[int] | None = None,
    ) -> pd.DataFrame:
        n = len(prices)
        volumes = volumes or [1.0] * n
        tick_rules = tick_rules or [1] * n
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        return pd.DataFrame(
            {
                "timestamp": [base_time + pd.Timedelta(seconds=i) for i in range(n)],
                "price": [Decimal(str(price)) for price in prices],
                "volume": [Decimal(str(volume)) for volume in volumes],
                "notional": [
                    Decimal(str(price * volume))
                    for price, volume in zip(prices, volumes, strict=True)
                ],
                "tickRule": tick_rules,
            }
        )

    def test_ohlc(self):
        df = self.get_data_frame(prices=[100, 105, 95, 102])
        result = aggregate_candle(df)

        self.assertEqual(result["open"], Decimal(100))
        self.assertEqual(result["high"], Decimal(105))
        self.assertEqual(result["low"], Decimal(95))
        self.assertEqual(result["close"], Decimal(102))

    def test_volume_aggregation(self):
        df = self.get_data_frame(prices=[100, 100], volumes=[5, 10])
        result = aggregate_candle(df)

        self.assertEqual(result["volume"], Decimal(15))
        self.assertEqual(result["notional"], Decimal(1500))

    def test_volume_aggregation_rejects_partial_totalized_schema(self):
        df = self.get_data_frame(prices=[100])
        df["totalVolume"] = Decimal(1)

        with self.assertRaisesRegex(ValueError, "totalNotional"):
            aggregate_candle(df)

    def test_buy_volume_aggregation(self):
        df = self.get_data_frame(
            prices=[100, 100, 100],
            volumes=[5, 10, 3],
            tick_rules=[1, -1, 1],
        )
        result = aggregate_candle(df)

        self.assertEqual(result["volume"], Decimal(18))
        self.assertEqual(result["buyVolume"], Decimal(8))
        self.assertEqual(result["buyNotional"], Decimal(800))

    def test_ticks_count(self):
        df = self.get_data_frame(prices=[100, 100, 100], tick_rules=[1, -1, 1])
        result = aggregate_candle(df)

        self.assertEqual(result["ticks"], 3)
        self.assertEqual(result["buyTicks"], 2)

    def test_timestamp_from_first_row(self):
        df = self.get_data_frame(prices=[100, 101])
        result = aggregate_candle(df)

        self.assertEqual(result["timestamp"], datetime(2024, 1, 1, tzinfo=UTC))

    def test_timestamp_override(self):
        df = self.get_data_frame(prices=[100, 101])
        custom_ts = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
        result = aggregate_candle(df, timestamp=custom_ts)

        self.assertEqual(result["timestamp"], custom_ts)

    def test_realized_variance_single_trade(self):
        df = self.get_data_frame(prices=[100])
        result = aggregate_candle(df)

        self.assertEqual(result["realizedVariance"], Decimal(0))

    def test_realized_variance_multiple_trades(self):
        df = self.get_data_frame(prices=[100, 110, 105])
        result = aggregate_candle(df)

        self.assertGreater(result["realizedVariance"], Decimal(0))

    def test_realized_variance_ignores_non_positive_prices(self):
        df = self.get_data_frame(prices=[100, 0, 110])
        result = aggregate_candle(df)

        self.assertGreater(result["realizedVariance"], Decimal(0))

    def test_round_volume_filtering(self):
        df = self.get_data_frame(
            prices=[100, 100],
            volumes=[1000, 5],
            tick_rules=[1, 1],
        )
        result = aggregate_candle(df, min_volume_exponent=1)

        self.assertEqual(result["roundVolume"], Decimal(1000))
        self.assertEqual(result["roundBuyVolume"], Decimal(1000))
        self.assertEqual(
            result["roundVolumeSumNotional"],
            Decimal(100000),
        )
        self.assertEqual(
            result["roundBuyVolumeSumNotional"],
            Decimal(100000),
        )

    def test_round_notional_filtering(self):
        df = self.get_data_frame(
            prices=[100, 100, 101],
            volumes=[4.5, 1.0, 1.0],
            tick_rules=[1, -1, 1],
        )
        result = aggregate_candle(df, min_notional_exponent=1)

        self.assertEqual(result["roundNotional"], Decimal("651.0"))
        self.assertEqual(result["roundBuyNotional"], Decimal("551.0"))
        self.assertEqual(
            result["roundNotionalSumVolume"],
            Decimal("6.5"),
        )
        self.assertEqual(
            result["roundBuyNotionalSumVolume"],
            Decimal("5.5"),
        )

    def test_round_stats_default_off(self):
        df = self.get_data_frame(prices=[100, 100], volumes=[100, 5])
        result = aggregate_candle(df)

        self.assertNotIn("roundVolume", result)
        self.assertNotIn("roundNotional", result)
        self.assertNotIn("volumeBrackets", result)

    def test_volume_brackets_use_exact_exclusive_upper_bounds(self):
        bounds = [10000, 100000, 1000000, 10000000]
        values = [Decimal("9999.999999999999999999"), *map(Decimal, bounds)]
        df = self.get_data_frame([Decimal(1000)] * 5, values, [1, -1, 1, -1, 1])
        df["notional"] = df.volume / Decimal(1000)
        df["ticks"] = [2, 3, 4, 5, 6]
        ordinary = aggregate_candle(df)
        result = aggregate_candle(df, volume_brackets=bounds)
        brackets = result.pop("volumeBrackets")
        self.assertEqual(result, ordinary)
        self.assertEqual(aggregate_candle(df, volume_brackets=[]), ordinary)
        for i, bucket in enumerate(brackets["buckets"]):
            self.assertEqual(bucket["volume"], values[i])
            self.assertEqual(bucket["buyVolume"], values[i] if i % 2 == 0 else 0)
            self.assertEqual(bucket["notional"], values[i] / 1000)
            self.assertEqual(bucket["ticks"], i + 2)
            self.assertEqual(bucket["minVolume"], 0 if i == 0 else bounds[i - 1])
            self.assertEqual(bucket["maxVolume"], bounds[i] if i < 4 else None)
        self.assertTrue(all(value == 0 for value in brackets["unclassified"].values()))

    def test_filtered_volume_brackets_preserve_sides_and_cache_remainders(self):
        df = self.get_data_frame(
            [Decimal(1000)] * 6,
            [900, 900, 1500, 2000, 100, 200],
            [-1, -1, 1, -1, 1, -1],
        )
        df["notional"] = df.volume / Decimal(1000)
        df["uid"], df["nanoseconds"], df["ticks"] = range(len(df)), 0, 1
        filtered = volume_filter_with_time_window(df, min_volume=1000, window=None)
        expected = aggregate_candle(filtered, volume_brackets=[1000, 2000])
        brackets = expected["volumeBrackets"]
        self.assertEqual(
            [bucket["volume"] for bucket in brackets["buckets"]], [0, 1500, 2000]
        )
        self.assertEqual(
            [bucket["buyVolume"] for bucket in brackets["buckets"]], [0, 1500, 0]
        )
        self.assertEqual(brackets["unclassified"]["volume"], 2100)
        self.assertEqual(brackets["unclassified"]["buyVolume"], 100)
        self.assertEqual(brackets["unclassified"]["ticks"], 4)
        self.assertEqual(brackets["unclassified"]["buyTicks"], 1)
        for field, remainder in brackets["unclassified"].items():
            self.assertEqual(
                remainder + sum(bucket[field] for bucket in brackets["buckets"]),
                expected[field],
            )
        for split in range(1, len(filtered)):
            with self.subTest(split=split):
                cache = get_next_cache(
                    filtered.iloc[:split], {}, volume_brackets=[1000, 2000]
                )
                cache = json.loads(
                    json.dumps(cache, cls=QuantTickEncoder), cls=QuantTickDecoder
                )
                result = aggregate_candle(
                    filtered.iloc[split:], volume_brackets=[1000, 2000]
                )
                self.assertEqual(merge_cache(cache["next"], result), expected)
        corrupted = filtered.copy()
        corrupted.loc[0, "totalBuyVolume"] = 0
        with self.assertRaisesRegex(ValueError, "reconcile"):
            aggregate_candle(corrupted, volume_brackets=[1000, 2000])

    def test_volume_brackets_reject_invalid_configuration_and_mixed_cache(self):
        df = self.get_data_frame([100, 100])
        for bounds in (
            [0],
            [-1],
            [10, 10],
            [20, 10],
            [True],
            [None],
            ["NaN"],
            ["Infinity"],
            "10000",
            False,
        ):
            with (
                self.subTest(bounds=bounds),
                self.assertRaises((TypeError, ValueError)),
            ):
                aggregate_candle(df, volume_brackets=bounds)
        ordinary = aggregate_candle(df)
        first = aggregate_candle(df, volume_brackets=[10])
        second = aggregate_candle(df, volume_brackets=[20])
        for previous, current in (
            (ordinary, first),
            (first, ordinary),
            (first, second),
        ):
            with (
                self.subTest(previous=previous),
                self.assertRaisesRegex(ValueError, "volume_brackets changed"),
            ):
                merge_cache(deepcopy(previous), deepcopy(current))


class ResampleCandleTest(SimpleTestCase):
    def get_hourly_candles(
        self,
        timestamp_from: datetime,
        hours: list[int],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "timestamp": timestamp_from + pd.Timedelta(f"{hour}h"),
                    "open": Decimal(str(hour + 1)),
                    "high": Decimal(str(hour + 11)),
                    "low": Decimal(str(hour)),
                    "close": Decimal(str(hour + 2)),
                    "notional": Decimal(10),
                }
                for hour in hours
            ]
        ).set_index("timestamp")

    def test_resample_candles_aggregates_complete_source_window(self):
        timestamp_from = datetime(2026, 4, 1, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 1, 8, tzinfo=UTC)
        data_frame = self.get_hourly_candles(timestamp_from, list(range(8)))

        result = resample_candles(
            data_frame,
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            resolution_minutes=480,
            source_resolution_minutes=60,
        )

        self.assertEqual(list(result.index), [timestamp_from])
        candle = result.iloc[0]
        self.assertEqual(candle.open, Decimal(1))
        self.assertEqual(candle.high, Decimal(18))
        self.assertEqual(candle.low, Decimal(0))
        self.assertEqual(candle.close, Decimal(9))
        self.assertEqual(candle.notional, Decimal(80))

    def test_resample_candles_skips_incomplete_source_window(self):
        timestamp_from = datetime(2026, 4, 1, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 1, 8, tzinfo=UTC)
        data_frame = self.get_hourly_candles(timestamp_from, [0, 1, 2, 4, 5, 6, 7])

        result = resample_candles(
            data_frame,
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            resolution_minutes=480,
            source_resolution_minutes=60,
        )

        self.assertTrue(result.empty)

    def test_resample_candles_keeps_complete_windows_and_skips_partial_tail(self):
        timestamp_from = datetime(2026, 4, 1, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 1, 16, tzinfo=UTC)
        data_frame = self.get_hourly_candles(timestamp_from, list(range(8)) + [8, 9])

        result = resample_candles(
            data_frame,
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            resolution_minutes=480,
            source_resolution_minutes=60,
        )

        self.assertEqual(list(result.index), [timestamp_from])

    def test_zero_trade_source_omissions_preserve_only_elapsed_traded_windows(self):
        start = datetime(2026, 4, 1, tzinfo=UTC)
        frame = self.get_hourly_candles(start, [4, 12, 16, 32])
        result = resample_candles(
            frame,
            timestamp_from=start + pd.Timedelta("4h"),
            timestamp_to=start + pd.Timedelta("36h"),
            resolution_minutes=480,
            source_resolution_minutes=240,
            missing_candles_are_zero=True,
        )
        self.assertEqual(
            list(result.index),
            [start + pd.Timedelta("8h"), start + pd.Timedelta("16h")],
        )
        self.assertEqual(result.open.tolist(), [Decimal(13), Decimal(17)])
        self.assertEqual(result.close.tolist(), [Decimal(14), Decimal(18)])
        self.assertEqual(result.notional.tolist(), [Decimal(10), Decimal(10)])


class MergeCacheTest(SimpleTestCase):
    def get_candle_data(
        self,
        open_price: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        timestamp: datetime | None = None,
    ) -> dict:
        return {
            "timestamp": timestamp or datetime(2024, 1, 1, tzinfo=UTC),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": Decimal(10),
            "buyVolume": Decimal(6),
            "notional": Decimal(1000),
            "buyNotional": Decimal(600),
            "ticks": 5,
            "buyTicks": 3,
            "roundVolume": Decimal(10),
            "roundBuyVolume": Decimal(6),
            "roundVolumeSumNotional": Decimal(1000),
            "roundBuyVolumeSumNotional": Decimal(600),
            "roundNotional": Decimal(1000),
            "roundBuyNotional": Decimal(600),
            "roundNotionalSumVolume": Decimal(10),
            "roundBuyNotionalSumVolume": Decimal(6),
            "realizedVariance": Decimal("0.001"),
        }

    def test_merge_preserves_open(self):
        previous = self.get_candle_data(
            Decimal(100), Decimal(105), Decimal(98), Decimal(102)
        )
        current = self.get_candle_data(
            Decimal(102),
            Decimal(110),
            Decimal(101),
            Decimal(108),
            timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        )

        result = merge_cache(previous, current)

        self.assertEqual(result["open"], Decimal(100))
        self.assertEqual(result["timestamp"], datetime(2024, 1, 1, tzinfo=UTC))

    def test_merge_high_low(self):
        previous = self.get_candle_data(
            Decimal(100), Decimal(105), Decimal(98), Decimal(102)
        )
        current = self.get_candle_data(
            Decimal(102),
            Decimal(110),
            Decimal(95),
            Decimal(108),
            timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        )

        result = merge_cache(previous, current)

        self.assertEqual(result["high"], Decimal(110))
        self.assertEqual(result["low"], Decimal(95))

    def test_merge_sums(self):
        previous = self.get_candle_data(
            Decimal(100), Decimal(105), Decimal(98), Decimal(102)
        )
        current = self.get_candle_data(
            Decimal(102),
            Decimal(110),
            Decimal(101),
            Decimal(108),
            timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        )

        result = merge_cache(previous, current)

        self.assertEqual(result["volume"], Decimal(20))
        self.assertEqual(result["buyVolume"], Decimal(12))
        self.assertEqual(result["notional"], Decimal(2000))
        self.assertEqual(result["buyNotional"], Decimal(1200))
        self.assertEqual(result["ticks"], 10)
        self.assertEqual(result["buyTicks"], 6)
        self.assertEqual(result["roundVolumeSumNotional"], Decimal(2000))
        self.assertEqual(
            result["roundBuyNotionalSumVolume"],
            Decimal(12),
        )

    def test_merge_realized_variance_with_cross_segment(self):
        previous = self.get_candle_data(
            Decimal(100), Decimal(105), Decimal(98), Decimal(102)
        )
        current = self.get_candle_data(
            Decimal(103),
            Decimal(110),
            Decimal(101),
            Decimal(108),
            timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        )

        result = merge_cache(previous, current)

        expected_min = Decimal("0.001") + Decimal("0.001")
        self.assertGreater(result["realizedVariance"], expected_min)

    def test_merge_without_round_keys(self):
        previous = self.get_candle_data(
            Decimal(100), Decimal(105), Decimal(98), Decimal(102)
        )
        current = self.get_candle_data(
            Decimal(102),
            Decimal(110),
            Decimal(101),
            Decimal(108),
            timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        )
        for key in (
            "roundVolume",
            "roundBuyVolume",
            "roundVolumeSumNotional",
            "roundBuyVolumeSumNotional",
            "roundNotional",
            "roundBuyNotional",
            "roundNotionalSumVolume",
            "roundBuyNotionalSumVolume",
        ):
            previous.pop(key)
            current.pop(key)

        result = merge_cache(previous, current)

        self.assertNotIn("roundVolume", result)
        self.assertNotIn("roundNotional", result)
        self.assertEqual(result["volume"], Decimal(20))
