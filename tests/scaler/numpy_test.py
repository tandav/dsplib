import numpy as np
import pytest

from dsplib.scale import minmax_scaler  # type: ignore


@pytest.mark.parametrize(
    'value, oldmin, oldmax, newmin, newmax', [
        (np.linspace(0, 5), 0, 2, 0, 1),
    ],
)
def test_value_inside_bounds(value, oldmin, oldmax, newmin, newmax):
    with pytest.raises(ValueError):
        minmax_scaler(value, oldmin, oldmax, newmin, newmax)


@pytest.mark.parametrize(
    'a, expected', [
        (np.linspace(0, 100, 11), np.linspace(0, 1, 11)),
        (np.linspace(100, 0, 11), np.linspace(1, 0, 11)),
    ],
)
def test_minmax_scaler_array(a, expected):
    assert np.allclose(minmax_scaler(a, np.min(a), np.max(a)), expected)


def test_np_dtype_overflow_check():
    a = np.array([0, 100, -120], dtype=np.int8)
    with pytest.raises(FloatingPointError):
        minmax_scaler(a, np.min(a), np.max(a))
