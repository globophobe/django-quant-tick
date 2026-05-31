from decimal import Decimal
from unittest import TestCase

import pandas as pd

from quant_tick.lib import get_frame_totals, has_column_group, is_decimal_close, validate_totals


class DataFrameValidationTest(TestCase):
    def test_is_decimal_close_uses_absolute_epsilon_only(self):
        self.assertTrue(is_decimal_close(Decimal("1.000000001"), Decimal("1.000000002")))
        self.assertFalse(is_decimal_close(Decimal("100000"), Decimal("100001")))
        self.assertFalse(is_decimal_close(Decimal("100"), Decimal("100.001")))

    def test_has_column_group_rejects_partial_groups(self):
        data = pd.DataFrame([{"a": 1, "b": 2}])

        with self.assertRaisesRegex(ValueError, "c"):
            has_column_group(data, ("a", "b", "c"))

    def test_get_frame_totals_uses_filtered_trade_bucket_totals(self):
        data = pd.DataFrame(
            [
                {
                    "volume": Decimal("999"),
                    "notional": Decimal("9.99"),
                    "totalVolume": Decimal("1000"),
                    "totalNotional": Decimal("10"),
                },
                {
                    "volume": Decimal("1"),
                    "notional": Decimal("0.01"),
                    "totalVolume": Decimal("2000"),
                    "totalNotional": Decimal("20"),
                },
            ]
        )

        volume, notional = get_frame_totals(data)

        self.assertEqual(volume, Decimal("3000"))
        self.assertEqual(notional, Decimal("30"))

    def test_get_frame_totals_rejects_partial_filtered_trade_totals(self):
        data = pd.DataFrame(
            [
                {
                    "volume": Decimal("3000"),
                    "notional": Decimal("30"),
                    "totalVolume": Decimal("3000"),
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "totalNotional"):
            get_frame_totals(data)

    def test_validate_totals_compares_raw_and_filtered_trade_totals(self):
        raw = pd.DataFrame(
            [
                {
                    "volume": Decimal("3000"),
                    "notional": Decimal("30"),
                }
            ]
        )
        totalized = pd.DataFrame(
            [
                {
                    "totalVolume": Decimal("3000"),
                    "totalNotional": Decimal("30"),
                }
            ]
        )

        validate_totals(raw=raw, totalized=totalized)

    def test_validate_totals_rejects_mismatched_totals(self):
        raw = pd.DataFrame([{"volume": Decimal("3000"), "notional": Decimal("30")}])
        totalized = pd.DataFrame(
            [{"totalVolume": Decimal("3000"), "totalNotional": Decimal("31")}]
        )

        with self.assertRaisesRegex(ValueError, "totalized notional"):
            validate_totals(raw=raw, totalized=totalized)
