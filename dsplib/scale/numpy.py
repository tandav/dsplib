import typing as tp

import numpy as np


def _within_bounds_array(
    value: np.ndarray,
    min_: float,
    max_: float,
) -> bool:
    if min_ == max_:
        return bool(np.all(value == min_))
    min_, max_ = min(min_, max_), max(min_, max_)
    return bool(np.all((min_ <= value) & (value <= max_)))


def minmax_scaler_array(
    value: np.ndarray,
    oldmin: tp.Optional[float] = None,
    oldmax: tp.Optional[float] = None,
    newmin: float = 0.0,
    newmax: float = 1.0,
) -> np.ndarray:

    if oldmin is None:
        oldmin = np.min(value)

    if oldmax is None:
        oldmax = np.max(value)

    if not _within_bounds_array(value, oldmin, oldmax):
        raise ValueError('value should be oldmin <= value <= oldmax')

    if oldmin == oldmax:
        if newmin != newmax:
            raise ValueError('oldmin == oldmax, so newmin == newmax must be true')
        return newmin

    old_settings = np.seterr(over='raise')
    out = (value - oldmin) * (newmax - newmin) / (oldmax - oldmin) + newmin
    np.seterr(**old_settings)
    return out


def rescale_around_zero(
    a: np.ndarray,
    oldmin: tp.Optional[float] = None,
    oldmax: tp.Optional[float] = None,
    newmin: float = -1,
    newmax: float = 1,
    zero: float = 0,
) -> np.ndarray:
    """
    Rescale audio signal to a new range while preserving a specific zero point.
    """
    # Shift the signal so that 'zero' is at 0
    shifted_signal = a - zero

    # Separate positive and negative values
    positive = np.maximum(shifted_signal, 0)
    negative = np.minimum(shifted_signal, 0)

    # Compute oldmin and oldmax if not provided
    oldmin = np.min(negative) if oldmin is None else oldmin - zero
    oldmax = np.max(positive) if oldmax is None else oldmax - zero

    # Rescale positive values
    if oldmax != 0:
        positive = positive / oldmax * (newmax - zero)

    # Rescale negative values
    if oldmin != 0:
        negative = negative / np.abs(oldmin) * (zero - newmin)

    # Combine rescaled positive and negative values and shift back
    rescaled = positive + negative + zero

    return rescaled
