
from decimal import Decimal

from django.test import SimpleTestCase

from quant_tick.lib import calc_notional_exponent, calc_volume_exponent


class ExperimentalTest(SimpleTestCase):
    """Tests for current exponent-based roundness helpers."""

    def test_calc_volume_exponent_current_behavior(self):
        self.assertEqual(calc_volume_exponent(Decimal('10')), 1)
        self.assertEqual(calc_volume_exponent(Decimal('20')), 1)
        self.assertEqual(calc_volume_exponent(Decimal('100')), 2)
        self.assertEqual(calc_volume_exponent(Decimal('400')), 2)
        self.assertEqual(calc_volume_exponent(Decimal('450')), 1)
        self.assertEqual(calc_volume_exponent(Decimal('0.1')), 0)

    def test_calc_notional_exponent_current_behavior(self):
        self.assertEqual(calc_notional_exponent(Decimal('10')), 3)
        self.assertEqual(calc_notional_exponent(Decimal('20')), 3)
        self.assertEqual(calc_notional_exponent(Decimal('100')), 4)
        self.assertEqual(calc_notional_exponent(Decimal('400')), 4)
        self.assertEqual(calc_notional_exponent(Decimal('450')), 3)
        self.assertEqual(calc_notional_exponent(Decimal('10.5')), 1)
        self.assertEqual(calc_notional_exponent(Decimal('450.25')), 0)
