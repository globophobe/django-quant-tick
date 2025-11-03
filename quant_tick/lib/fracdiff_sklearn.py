"""Fractional differentiation transformer.

Copyright (c) 2019-2022, Y. Nakaji, https://github.com/fracdiff/fracdiff
BSD 3-Clause License

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from typing import TypeVar

import numpy as np
from scipy.special import binom
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


def fdiff_coef(d: float, window: int) -> np.ndarray:
    """Returns sequence of coefficients in fracdiff operator.

    Parameters
    ----------
    d : float
        Order of differentiation.
    window : int
        Number of terms.

    Returns
    -------
    coef : numpy.array, shape (window,)
        Coefficients in fracdiff operator.

    Examples
    --------
    >>> fdiff_coef(0.5, 4)
    array([ 1.    , -0.5   , -0.125 , -0.0625])
    >>> fdiff_coef(1.0, 4)
    array([ 1., -1.,  0., -0.])
    >>> fdiff_coef(1.5, 4)
    array([ 1.    , -1.5   ,  0.375 ,  0.0625])
    """
    return (-1) ** np.arange(window) * binom(d, np.arange(window))


def fdiff(
    a: np.ndarray,
    n: float = 1.0,
    axis: int = -1,
    prepend: np.ndarray | None = None,
    append: np.ndarray | None = None,
    window: int = 10,
    mode: str = "same",
) -> np.ndarray:
    """Calculate the n-th differentiation along the given axis.

    Extension of numpy.diff to fractional differentiation.

    Parameters
    ----------
    a : array_like
        The input array.
    n : float, default=1.0
        The order of differentiation.
        If n is an integer, returns the same output with numpy.diff.
    axis : int, default=-1
        The axis along which the difference is taken.
    prepend, append : array_like, optional
        Values to prepend or append to a along axis prior to performing the
        difference.
    window : int, default=10
        Number of observations to compute each element in the output.
    mode : str, default='same'
        'same' or 'valid'.
        If 'same', the output has the same length with ``a``.
        If 'valid', the output is shrinked so that boundary effects are discarded.

    Returns
    -------
    fdiff : ndarray
        The `n`-th differentiation along the given axis.

    Examples
    --------
    >>> fdiff(np.array([1, 2, 4, 7, 0]), 0.5, window=3)
    array([  1.        ,   1.5       ,   2.875     ,   4.8828125 ,
            -4.26171875])
    >>> fdiff(np.array([1, 2, 4, 7, 0]), 1.0, window=3)
    array([ 1.,  1.,  2.,  3., -7.])
    >>> fdiff(np.array([1, 2, 4, 7, 0]), 2.0, window=3)
    array([ 1.,  0.,  1.,  1., -10.])
    """
    a = np.asarray(a)

    if prepend is not None:
        a = np.concatenate((np.asarray(prepend), a), axis=axis)

    if append is not None:
        a = np.concatenate((a, np.asarray(append)), axis=axis)

    shape = a.shape
    a = np.moveaxis(a, axis, -1)
    a = a.reshape(-1, shape[axis])

    coef = fdiff_coef(n, window)

    out = []
    for i in range(a.shape[0]):
        row = a[i]
        result = np.convolve(row, coef, mode=mode)
        out.append(result)

    out = np.array(out)

    if mode == "valid":
        new_shape = list(shape)
        new_shape[axis] = out.shape[1]
        out = out.reshape(new_shape)
        out = np.moveaxis(out, -1, axis)
    else:
        out = out.reshape(shape)
        out = np.moveaxis(out, -1, axis)

    return out


T = TypeVar("T", bound="Fracdiff")


class Fracdiff(TransformerMixin, BaseEstimator):
    """A scikit-learn transformer to compute fractional differentiation."""

    def __init__(
        self,
        d: float = 1.0,
        window: int = 10,
        mode: str = "same",
        window_policy: str = "fixed",
    ) -> None:
        """Initialize fractional differencing transformer."""
        self.d = d
        self.window = window
        self.mode = mode
        self.window_policy = window_policy

    def __repr__(self) -> str:
        """String representation."""
        name = self.__class__.__name__
        attrs = ["d", "window", "mode", "window_policy"]
        params = ", ".join(f"{attr}={getattr(self, attr)}" for attr in attrs)
        return f"{name}({params})"

    def fit(self: T, X: np.ndarray, y: None = None) -> T:
        """Fit transformer."""
        self.coef_ = fdiff_coef(self.d, self.window)
        return self

    def transform(self, X: np.ndarray, y: None = None) -> np.ndarray:
        """Transform data."""
        check_is_fitted(self, ["coef_"])
        check_array(X, estimator=self)
        return fdiff(X, n=self.d, axis=0, window=self.window, mode=self.mode)
