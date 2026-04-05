from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.lib import merge_cache
from quant_tick.lib.candles import aggregate_candle


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

        self.assertEqual(result["open"], Decimal("100"))
        self.assertEqual(result["high"], Decimal("105"))
        self.assertEqual(result["low"], Decimal("95"))
        self.assertEqual(result["close"], Decimal("102"))

    def test_volume_aggregation(self):
        df = self.get_data_frame(prices=[100, 100], volumes=[5, 10])
        result = aggregate_candle(df)

        self.assertEqual(result["volume"], Decimal("15"))
        self.assertEqual(result["notional"], Decimal("1500"))

    def test_buy_volume_aggregation(self):
        df = self.get_data_frame(
            prices=[100, 100, 100],
            volumes=[5, 10, 3],
            tick_rules=[1, -1, 1],
        )
        result = aggregate_candle(df)

        self.assertEqual(result["volume"], Decimal("18"))
        self.assertEqual(result["buyVolume"], Decimal("8"))
        self.assertEqual(result["buyNotional"], Decimal("800"))

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

        self.assertEqual(result["realizedVariance"], Decimal("0"))

    def test_realized_variance_multiple_trades(self):
        df = self.get_data_frame(prices=[100, 110, 105])
        result = aggregate_candle(df)

        self.assertGreater(result["realizedVariance"], Decimal("0"))

    def test_round_volume_filtering(self):
        df = self.get_data_frame(
            prices=[100, 100],
            volumes=[100, 5],
            tick_rules=[1, 1],
        )
        result = aggregate_candle(df, min_volume_exponent=2, round_volume=True)

        self.assertEqual(result["roundVolume"], Decimal("100"))
        self.assertEqual(result["roundBuyVolume"], Decimal("100"))
        self.assertEqual(
            result["roundVolumeSumNotional"],
            Decimal("10000"),
        )
        self.assertEqual(
            result["roundBuyVolumeSumNotional"],
            Decimal("10000"),
        )

    def test_round_notional_filtering(self):
        df = self.get_data_frame(
            prices=[100, 100, 101],
            volumes=[4.5, 1.0, 1.0],
            tick_rules=[1, -1, 1],
        )
        result = aggregate_candle(df, round_notional=True)

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
            "volume": Decimal("10"),
            "buyVolume": Decimal("6"),
            "notional": Decimal("1000"),
            "buyNotional": Decimal("600"),
            "ticks": 5,
            "buyTicks": 3,
            "roundVolume": Decimal("10"),
            "roundBuyVolume": Decimal("6"),
            "roundVolumeSumNotional": Decimal("1000"),
            "roundBuyVolumeSumNotional": Decimal("600"),
            "roundNotional": Decimal("1000"),
            "roundBuyNotional": Decimal("600"),
            "roundNotionalSumVolume": Decimal("10"),
            "roundBuyNotionalSumVolume": Decimal("6"),
            "realizedVariance": Decimal("0.001"),
        }

    def test_merge_preserves_open(self):
        previous = self.get_candle_data(
            Decimal("100"), Decimal("105"), Decimal("98"), Decimal("102")
        )
        current = self.get_candle_data(
            Decimal("102"),
            Decimal("110"),
            Decimal("101"),
            Decimal("108"),
            timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        )

        result = merge_cache(previous, current)

        self.assertEqual(result["open"], Decimal("100"))
        self.assertEqual(result["timestamp"], datetime(2024, 1, 1, tzinfo=UTC))

    def test_merge_high_low(self):
        previous = self.get_candle_data(
            Decimal("100"), Decimal("105"), Decimal("98"), Decimal("102")
        )
        current = self.get_candle_data(
            Decimal("102"),
            Decimal("110"),
            Decimal("95"),
            Decimal("108"),
            timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        )

        result = merge_cache(previous, current)

        self.assertEqual(result["high"], Decimal("110"))
        self.assertEqual(result["low"], Decimal("95"))

    def test_merge_sums(self):
        previous = self.get_candle_data(
            Decimal("100"), Decimal("105"), Decimal("98"), Decimal("102")
        )
        current = self.get_candle_data(
            Decimal("102"),
            Decimal("110"),
            Decimal("101"),
            Decimal("108"),
            timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        )

        result = merge_cache(previous, current)

        self.assertEqual(result["volume"], Decimal("20"))
        self.assertEqual(result["buyVolume"], Decimal("12"))
        self.assertEqual(result["notional"], Decimal("2000"))
        self.assertEqual(result["buyNotional"], Decimal("1200"))
        self.assertEqual(result["ticks"], 10)
        self.assertEqual(result["buyTicks"], 6)
        self.assertEqual(result["roundVolumeSumNotional"], Decimal("2000"))
        self.assertEqual(
            result["roundBuyNotionalSumVolume"],
            Decimal("12"),
        )

    def test_merge_realized_variance_with_cross_segment(self):
        previous = self.get_candle_data(
            Decimal("100"), Decimal("105"), Decimal("98"), Decimal("102")
        )
        current = self.get_candle_data(
            Decimal("103"),
            Decimal("110"),
            Decimal("101"),
            Decimal("108"),
            timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        )

        result = merge_cache(previous, current)

        expected_min = Decimal("0.001") + Decimal("0.001")
        self.assertGreater(result["realizedVariance"], expected_min)

    def test_merge_without_round_keys(self):
        previous = self.get_candle_data(
            Decimal("100"), Decimal("105"), Decimal("98"), Decimal("102")
        )
        current = self.get_candle_data(
            Decimal("102"),
            Decimal("110"),
            Decimal("101"),
            Decimal("108"),
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
        self.assertEqual(result["volume"], Decimal("20"))
