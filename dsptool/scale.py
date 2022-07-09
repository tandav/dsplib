import numpy as np


def minmax_scaler(value, oldmin, oldmax, newmin=0.0, newmax=1.0) -> float:
    if not oldmin < oldmax:
        raise ValueError('oldmin should be less than oldmax')
    if not newmin < newmax:
        raise ValueError('newmin should be less than newmax')
    if not oldmin <= value <= oldmax:
        raise ValueError('value should be oldmin <= value <= oldmax')

    old_settings = np.seterr(over='raise')
    out = (value - oldmin) * (newmax - newmin) / (oldmax - oldmin) + newmin
    np.seterr(**old_settings)
    return out


def minmax_scaler_array(a: np.ndarray, newmin: float = 0, newmax: float = 1) -> np.ndarray:
    return np.array([minmax_scaler(x, min(a), max(a), newmin, newmax) for x in a])
