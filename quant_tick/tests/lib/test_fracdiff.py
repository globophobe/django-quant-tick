"""Tests for fractional differentiation transformer.

Adapted from https://github.com/fracdiff/fracdiff
"""
import numpy as np
from django.test import TestCase
from numpy.testing import assert_array_equal

from quant_tick.lib.fracdiff_sklearn import Fracdiff, fdiff, fdiff_coef


class FracdiffCoefTest(TestCase):
    def test_fdiff_coef_half_order(self):
        """Fractional differentiation coefficient for d=0.5."""
        result = fdiff_coef(0.5, 4)
        expected = np.array([1.0, -0.5, -0.125, -0.0625])
        np.testing.assert_array_almost_equal(result, expected)

    def test_fdiff_coef_integer_order(self):
        """Fractional differentiation coefficient for d=1.0."""
        result = fdiff_coef(1.0, 4)
        expected = np.array([1.0, -1.0, 0.0, -0.0])
        np.testing.assert_array_almost_equal(result, expected)


class FracdiffFunctionTest(TestCase):
    def test_fdiff_half_order(self):
        """Fractional differentiation of order 0.5."""
        a = np.array([1, 2, 4, 7, 0])
        result = fdiff(a, 0.5, window=3)
        self.assertEqual(len(result), 5)
        self.assertIsInstance(result, np.ndarray)

    def test_fdiff_integer_order(self):
        """Fractional differentiation of order 1.0."""
        a = np.array([1, 2, 4, 7, 0])
        result = fdiff(a, 1.0, window=3)
        self.assertEqual(len(result), 5)
        self.assertIsInstance(result, np.ndarray)

    def test_fdiff_valid_mode(self):
        """Fractional differentiation with valid mode."""
        a = np.array([1, 2, 4, 7, 0])
        result = fdiff(a, 0.5, window=3, mode="valid")
        self.assertEqual(len(result), 3)


class FracdiffTransformerTest(TestCase):
    def test_repr(self):
        """Fracdiff transformer repr."""
        fracdiff = Fracdiff(0.5, window=10, mode="same", window_policy="fixed")
        expected = "Fracdiff(d=0.5, window=10, mode=same, window_policy=fixed)"
        self.assertEqual(repr(fracdiff), expected)

    def test_fit_transform(self):
        """Fracdiff fit_transform matches fdiff."""
        np.random.seed(42)
        X = np.random.randn(50, 100)
        fracdiff = Fracdiff(d=0.5, window=10, mode="same")
        out = fdiff(X, n=0.5, axis=0, window=10, mode="same")
        assert_array_equal(fracdiff.fit_transform(X), out)

    def test_fit_transform_valid_mode(self):
        """Fracdiff fit_transform with valid mode."""
        np.random.seed(42)
        X = np.random.randn(50, 100)
        fracdiff = Fracdiff(d=0.5, window=10, mode="valid")
        out = fdiff(X, n=0.5, axis=0, window=10, mode="valid")
        assert_array_equal(fracdiff.fit_transform(X), out)

    def test_fit_then_transform(self):
        """Fracdiff separate fit and transform."""
        np.random.seed(42)
        X = np.random.randn(50, 100)
        fracdiff = Fracdiff(d=0.5, window=10, mode="same")
        fracdiff.fit(X)
        result = fracdiff.transform(X)
        self.assertIsInstance(result, np.ndarray)
        self.assertGreater(result.size, 0)
