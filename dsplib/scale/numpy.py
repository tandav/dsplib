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
