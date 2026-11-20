from decimal import Decimal as D

from django.test import SimpleTestCase

from quant_tick.exchanges.position_metrics import (
    PNL_FIELDS,
    add_realized_pnl,
    add_taker_fill,
    concentration_values,
    empty_flow_tiers,
    minute_open_interest,
    validate_realized_pnl,
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

    def test_profit_and_loss_are_preserved_when_the_net_cancels(self):
        row = dict.fromkeys(PNL_FIELDS, D(0))
        row.update(taker_realized_pnl=D(0), maker_realized_pnl=D(60))
        for before, after, pnl, taker in (
            (10, 5, 100, True),
            (-3, 0, -100, True),
            (-2, 1, 80, False),
            (3, -1, -20, False),
        ):
            add_realized_pnl(row, D(before), D(after), D(pnl), taker=taker)
        self.assertEqual(row["taker_long_realized_profit"], 100)
        self.assertEqual(row["taker_short_realized_loss"], 100)
        self.assertEqual(row["maker_short_realized_profit"], 80)
        self.assertEqual(row["maker_long_realized_loss"], 20)
        validate_realized_pnl(row)
        with self.assertRaisesRegex(ValueError, "closing"):
            add_realized_pnl(row, D(1), D(2), D(1), taker=True)
        row["maker_long_realized_loss"] += 1
        with self.assertRaisesRegex(ValueError, "reconcile"):
            validate_realized_pnl(row)

    def test_concentration_handles_empty_sides_and_rejects_inconsistent_counts(self):
        zero = {"accounts": 0, "size": D(0), "top1_size": D(0), "top5_size": D(0)}
        buy = {"accounts": 6, "size": D(9), "top1_size": D(3), "top5_size": D(8)}
        totals = {"taker_buy_size": D(9), "taker_sell_size": D(0)}
        row = concentration_values(6, {"buy": buy, "sell": zero}, totals)
        self.assertEqual(row["buy"]["top1_share"], D(1) / 3)
        self.assertEqual(row["buy"]["top5_share"], D(8) / 9)
        self.assertIsNone(row["sell"]["top1_share"])
        for change in (
            {"accounts": 2},
            {"top1_size": D(0)},
            {"top5_size": D(10)},
            {"size": D(10)},
            {"top1_size": D("NaN")},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                concentration_values(
                    6, {"buy": {**buy, **change}, "sell": zero}, totals
                )
