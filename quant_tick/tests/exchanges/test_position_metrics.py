from decimal import Decimal as D

from django.test import SimpleTestCase

from quant_tick.exchanges.position_metrics import (
    add_taker_fill,
    empty_flow_tiers,
    minute_open_interest,
)


class PositionMetricsTest(SimpleTestCase):
    def test_brackets_follow_pre_fill_positions_and_split_reversals(self):
        cases = (
            ("0", "200", "20000000", 1, "long_open", "200", None, "0"),
            ("100", "101", "100000", 5, "long_open", "1", None, "0"),
            ("-100", "-99", "100000", 5, "short_close", "1", None, "0"),
            ("-2", "1", "300000", 3, "short_close", "2", "long_open", "1"),
            ("2", "-1", "300000", 3, "long_close", "2", "short_open", "1"),
        )
        for before, after, value, tier, first, quantity, second, remainder in cases:
            with self.subTest(before=before, after=after):
                rows = empty_flow_tiers()
                add_taker_fill(rows, D(before), D(after), D(value))
                row = rows[tier - 1]
                self.assertEqual(row["taker_" + first + "_size"], D(quantity))
                if second:
                    self.assertEqual(row["taker_" + second + "_size"], D(remainder))
                self.assertEqual(
                    [
                        r["tier"]
                        for r in rows
                        if r["taker_buy_size"] or r["taker_sell_size"]
                    ],
                    [tier],
                )

    def test_minute_samples_preserve_unwinds_even_when_the_hour_finishes_higher(self):
        result = minute_open_interest(D(100), D(103), {3: D(10), 1: D(-20), 2: D(13)})
        expected = {
            "open": 100,
            "close": 103,
            "high": 103,
            "low": 80,
            "increase": 23,
            "decrease": 20,
            "max_drawdown": 20,
        }
        for field, value in expected.items():
            self.assertEqual(result["minute_open_interest_" + field], D(value))
        # Offset changes in one minute are not advertised as intraminute extrema.
        result = minute_open_interest(D(100), D(100), {1: D(0)})
        self.assertEqual(result["minute_open_interest_max_drawdown"], 0)
        self.assertEqual(result, minute_open_interest(D(100), D(100), {}))

    def test_missing_or_impossible_minute_changes_fail_reconciliation(self):
        for before, after, changes in (
            (D(100), D(103), {}),
            (D(100), D(103), {1: D(-101), 2: D(104)}),
            (D(100), D(103), {1: D("NaN")}),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                minute_open_interest(before, after, changes)
