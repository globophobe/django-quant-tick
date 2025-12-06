from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.lib import merge_cache
from quant_tick.lib.candles import agg_candle


class SingleExchangeCandleTest(SimpleTestCase):
    """Single exchange candle."""

    def get_data_frame(
        self,
        prices: list[float],
        volumes: list[float] | None = None,
        tick_rules: list[int] | None = None,
        exchange: str | None = None,
    ) -> pd.DataFrame:
        """Get data frame."""
        n = len(prices)
        volumes = volumes or [1.0] * n
        tick_rules = tick_rules or [1] * n
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

        data = {
            "timestamp": [base_time + pd.Timedelta(seconds=i) for i in range(n)],
            "price": [Decimal(str(p)) for p in prices],
            "volume": [Decimal(str(v)) for v in volumes],
            "notional": [Decimal(str(p * v)) for p, v in zip(prices, volumes, strict=True)],
            "tickRule": tick_rules,
        }
        if exchange:
            data["exchange"] = [exchange] * n
        return pd.DataFrame(data)

    def test_ohlc(self):
        """OHLC."""
        df = self.get_data_frame(prices=[100, 105, 95, 102])
        result = agg_candle(df)

        self.assertEqual(result["open"], Decimal("100"))
        self.assertEqual(result["high"], Decimal("105"))
        self.assertEqual(result["low"], Decimal("95"))
        self.assertEqual(result["close"], Decimal("102"))

    def test_volume_aggregation(self):
        """Volume aggregation."""
        df = self.get_data_frame(prices=[100, 100], volumes=[5, 10])
        result = agg_candle(df)

        self.assertEqual(result["volume"], Decimal("15"))
        self.assertEqual(result["notional"], Decimal("1500"))

    def test_buy_volume_aggregation(self):
        """Buy volume aggregation."""
        df = self.get_data_frame(
            prices=[100, 100, 100],
            volumes=[5, 10, 3],
            tick_rules=[1, -1, 1],
        )
        result = agg_candle(df)

        self.assertEqual(result["volume"], Decimal("18"))
        self.assertEqual(result["buyVolume"], Decimal("8"))  # 5 + 3

    def test_ticks_count(self):
        """Tick count."""
        df = self.get_data_frame(prices=[100, 100, 100], tick_rules=[1, -1, 1])
        result = agg_candle(df)

        self.assertEqual(result["ticks"], 3)
        self.assertEqual(result["buyTicks"], 2)

    def test_timestamp_from_first_row(self):
        """Timestamp from first row."""
        df = self.get_data_frame(prices=[100, 101])
        result = agg_candle(df)

        expected = datetime(2024, 1, 1, tzinfo=timezone.utc)
        self.assertEqual(result["timestamp"], expected)

    def test_timestamp_override(self):
        """Timestamp override."""
        df = self.get_data_frame(prices=[100, 101])
        custom_ts = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
        result = agg_candle(df, timestamp=custom_ts)

        self.assertEqual(result["timestamp"], custom_ts)

    def test_realized_variance_single_trade(self):
        """Realized variance single trade."""
        df = self.get_data_frame(prices=[100])
        result = agg_candle(df)

        self.assertEqual(result["realizedVariance"], Decimal("0"))

    def test_realized_variance_multiple_trades(self):
        """Realized variance multiple trades."""
        df = self.get_data_frame(prices=[100, 110, 105])
        result = agg_candle(df)

        self.assertGreater(result["realizedVariance"], Decimal("0"))

    def test_round_volume_filtering(self):
        """Round volume filtering."""
        df = self.get_data_frame(
            prices=[100, 100],
            volumes=[100, 5],  # 100 is round, 5 is not
            tick_rules=[1, 1],
        )
        result = agg_candle(df, min_volume_exponent=2)

        self.assertEqual(result["volume"], Decimal("105"))
        self.assertEqual(result["roundVolume"], Decimal("100"))




class MergeCacheTest(SimpleTestCase):
    """Merge cache."""

    def get_candle_data(
        self,
        open_price: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        timestamp: datetime | None = None,
    ) -> dict:
        """Create candle data for testing."""
        return {
            "timestamp": timestamp or datetime(2024, 1, 1, tzinfo=timezone.utc),
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
            "roundNotional": Decimal("1000"),
            "roundBuyNotional": Decimal("600"),
            "realizedVariance": Decimal("0.001"),
        }

    def test_merge_preserves_open(self):
        """Merged candle uses open from first."""
        previous = self.get_candle_data(
            Decimal("100"), Decimal("105"), Decimal("98"), Decimal("102")
        )
        current = self.get_candle_data(
            Decimal("102"), Decimal("110"), Decimal("101"), Decimal("108"),
            timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
        )

        result = merge_cache(previous, current)

        self.assertEqual(result["open"], Decimal("100"))
        self.assertEqual(result["timestamp"], datetime(2024, 1, 1, tzinfo=timezone.utc))

    def test_merge_high_low(self):
        """Merged candle uses max high and min low."""
        previous = self.get_candle_data(
            Decimal("100"), Decimal("105"), Decimal("98"), Decimal("102")
        )
        current = self.get_candle_data(
            Decimal("102"), Decimal("110"), Decimal("95"), Decimal("108"),
            timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
        )

        result = merge_cache(previous, current)

        self.assertEqual(result["high"], Decimal("110"))  # max(105, 110)
        self.assertEqual(result["low"], Decimal("95"))  # min(98, 95)

    def test_merge_sums(self):
        """Merge sums."""
        previous = self.get_candle_data(
            Decimal("100"), Decimal("105"), Decimal("98"), Decimal("102")
        )
        current = self.get_candle_data(
            Decimal("102"), Decimal("110"), Decimal("101"), Decimal("108"),
            timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
        )

        result = merge_cache(previous, current)

        self.assertEqual(result["volume"], Decimal("20"))
        self.assertEqual(result["buyVolume"], Decimal("12"))
        self.assertEqual(result["notional"], Decimal("2000"))
        self.assertEqual(result["ticks"], 10)

    def test_merge_realized_variance_with_cross_segment(self):
        """Realized variance with cross-segment."""
        previous = self.get_candle_data(
            Decimal("100"), Decimal("105"), Decimal("98"), Decimal("102")
        )
        current = self.get_candle_data(
            Decimal("103"), Decimal("110"), Decimal("101"), Decimal("108"),
            timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
        )

        result = merge_cache(previous, current)

        # Should be > sum of individual variances due to cross-segment
        expected_min = Decimal("0.001") + Decimal("0.001")
        self.assertGreater(result["realizedVariance"], expected_min)
