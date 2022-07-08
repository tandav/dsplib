import numpy as np


def minmax_scaler(value, oldmin, oldmax, newmin=0.0, newmax=1.0) -> float:
    return (value - oldmin) * (newmax - newmin) / (oldmax - oldmin) + newmin


def minmax_scaler_array(a: np.ndarray, newmin: float = 0, newmax: float = 1) -> np.ndarray:
    return np.array([minmax_scaler(x, min(a), max(a), newmin, newmax) for x in a])
