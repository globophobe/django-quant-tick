from decimal import Decimal

from django.test import SimpleTestCase

from quant_tick.lib.candles import is_round_notional, is_round_volume


class RoundnessTest(SimpleTestCase):

    def test_is_round_volume_current_threshold_behavior(self):
        self.assertFalse(is_round_volume(Decimal("10"), 1))
        self.assertFalse(is_round_volume(Decimal("100"), 1))
        self.assertTrue(is_round_volume(Decimal("1000"), 1))
        self.assertTrue(is_round_volume(Decimal("5000"), 1))
        self.assertTrue(is_round_volume(Decimal("10000"), 2))
        self.assertTrue(is_round_volume(Decimal("40000"), 2))
        self.assertFalse(is_round_volume(Decimal("40000"), 3))
        self.assertFalse(is_round_volume(Decimal("0.1"), 1))

    def test_is_round_notional_current_threshold_behavior(self):
        self.assertTrue(is_round_notional(Decimal("10"), 3))
        self.assertTrue(is_round_notional(Decimal("20"), 3))
        self.assertTrue(is_round_notional(Decimal("100"), 4))
        self.assertTrue(is_round_notional(Decimal("400"), 4))
        self.assertTrue(is_round_notional(Decimal("450"), 3))
        self.assertTrue(is_round_notional(Decimal("10.5"), 1))
        self.assertFalse(is_round_notional(Decimal("10.5"), 2))
        self.assertFalse(is_round_notional(Decimal("450.25"), 1))
