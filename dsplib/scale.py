import typing as tp

import numpy as np


def _within_bounds(
    value: tp.Union[float, np.ndarray],
    bounds: tp.Tuple[float, float],
) -> bool:
    a, b = bounds
    if a == b:
        return value == a
    min_, max_ = min(a, b), max(a, b)
    if isinstance(value, np.ndarray):
        return bool(np.all((min_ <= value) & (value <= max_)))
    return min_ <= value <= max_


def minmax_scaler(
    value: tp.Union[float, np.ndarray],
    oldmin: float,
    oldmax: float,
    newmin: float = 0.0,
    newmax: float = 1.0,
) -> tp.Union[float, np.ndarray]:

    if not _within_bounds(value, (oldmin, oldmax)):
        raise ValueError('value should be oldmin <= value <= oldmax')

    if oldmin == oldmax:
        if newmin != newmax:
            raise ValueError('oldmin == oldmax, so newmin == newmax must be true')
        return newmin

    old_settings = np.seterr(over='raise')
    out = (value - oldmin) * (newmax - newmin) / (oldmax - oldmin) + newmin
    np.seterr(**old_settings)
    return out
