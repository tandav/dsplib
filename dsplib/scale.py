import numpy as np


def minmax_scaler(
    value: float | np.ndarray,
    oldmin: float,
    oldmax: float,
    newmin: float = 0.0,
    newmax: float = 1.0,
) -> float | np.ndarray:
    if not oldmin < oldmax:
        raise ValueError('oldmin should be less than oldmax')
    if not newmin < newmax:
        raise ValueError('newmin should be less than newmax')

    if isinstance(value, np.ndarray):
        if not np.all((oldmin <= value) & (value <= oldmax)):
            raise ValueError('value should be oldmin <= value <= oldmax')
    else:
        if not oldmin <= value <= oldmax:
            raise ValueError('value should be oldmin <= value <= oldmax')

    old_settings = np.seterr(over='raise')
    out = (value - oldmin) * (newmax - newmin) / (oldmax - oldmin) + newmin
    np.seterr(**old_settings)
    return out
