import numpy as np
import pytest

from dsplib.scale import minmax_scaler  # type: ignore
from dsplib.scale import minmax_scaler_array  # type: ignore
from dsplib.scale.numpy import rescale_around_zero


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
    assert np.allclose(minmax_scaler_array(a, np.min(a), np.max(a)), expected)
    assert np.allclose(minmax_scaler_array(a), expected)


def test_np_dtype_overflow_check():
    a = np.array([0, 100, -120], dtype=np.int8)
    with pytest.raises(FloatingPointError):
        minmax_scaler_array(a, np.min(a), np.max(a))


@pytest.mark.parametrize(
    'a, oldmin, oldmax, newmin, newmax, zero, expected', [
        ([-10, -8, -3, 0, 5, 10], None, None, -1, 1, 0, [-1, -0.8, -0.3, 0, 0.5, 1]),
        ([-100, -80, -30, 0, 50, 100], None, None, -1, 1, 0, [-1, -0.8, -0.3, 0, 0.5, 1]),
        ([-100, -80, -30, 0, 5, 10], None, None, -1, 1, 0, [-1, -0.8, -0.3, 0, 0.5, 1]),
        ([0, 1, 2, 3, 4, 5], None, None, 0, 10, 0, [0, 2, 4, 6, 8, 10]),
        ([-5, -3, 0, 5, 10], None, None, -1, 1, 0, [-1, -0.6, 0, 0.5, 1]),
        ([-5, -3, -2, -1, 0, 1, 2, 3, 5], None, None, -2, 2, 0, [-2, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 2]),
        ([12, 10, 8, 6, 4, 2, 0], None, None, -1, 1, 0, [1, 0.8333333, 0.6666667, 0.5, 0.3333333, 0.1666667, 0]),
        ([-2, -1, 0, 1, 2], -2, 2, -20, 20, 0, [-20, -10, 0, 10, 20]),
        ([-5, -2.5, 0, 2.5, 5], -5, 5, -1, 1, 0, [-1, -0.5, 0, 0.5, 1]),
    ],
)
def test_rescale_around_zero(a, oldmin, oldmax, newmin, newmax, zero, expected):
    a = np.array(a, dtype=np.float32)
    assert np.allclose(
        rescale_around_zero(a, oldmin, oldmax, newmin, newmax, zero),
        expected,
    )
