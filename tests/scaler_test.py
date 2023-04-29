import numpy as np
import pytest

from dsplib.scale import minmax_scaler


@pytest.mark.parametrize(
    'value, oldmin, oldmax, newmin, newmax, expected', [
        (0, 0, 0, 0, 0, 0),
        (3, 0, 10, 0, 0, 0),
        (3, 10, 0, 0, 0, 0),
        (0, 0, 10, 0, 1, 0),
        (10, 0, 10, 0, 1, 1),
        (10, 10, 0, 0, 1, 0),
        (7, 10, 0, 0, 1, 0.3),
        (7, 0, 10, 0, 1, 0.7),
        (3, 10, 0, 0, 1, 0.7),
        (0, 10, 0, 0, 1, 1),
        (0, 0, -1, 0, 1, 0),
        (0, 10, 10, 0, 1, ValueError),
        (0, 10, 9, 0, 1, ValueError),  # value should be oldmin <= value <= oldmax

        # # newmin, newmax
        (50, 0, 100, 0, -1, -0.5),
        (50, 0, 100, 10, 9, 9.5),
    ],
)
def test_min_less_than_max(value, oldmin, oldmax, newmin, newmax, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            minmax_scaler(value, oldmin, oldmax, newmin, newmax)
        return
    assert minmax_scaler(value, oldmin, oldmax, newmin, newmax) == expected


@pytest.mark.parametrize(
    'value, oldmin, oldmax, newmin, newmax', [
        (101, 0, 100, 0.0, 1.0),
        (-1, 0, 100, 0.0, 1.0),
        (np.linspace(0, 5), 0, 2, 0, 1),
    ],
)
def test_value_inside_bounds(value, oldmin, oldmax, newmin, newmax):
    with pytest.raises(ValueError):
        minmax_scaler(value, oldmin, oldmax, newmin, newmax)


@pytest.mark.parametrize('swap_old_new', [False, True])
@pytest.mark.parametrize(
    'value, oldmin, oldmax, newmin, newmax, expected', [
        pytest.param(-18, -90, -10, 0, 1, 0.9, id='--'),
        pytest.param(-10, -100, 0, 0, 1, 0.9, id='-0'),
        pytest.param(80, -10, 90, 0, 1, 0.9, id='-+'),
        pytest.param(90, 0, 100, 0, 1, 0.9, id='0+'),
        pytest.param(82, 10, 90, 0, 1, 0.9, id='++'),
    ],
)
def test_minmax_scaler(value, oldmin, oldmax, newmin, newmax, expected, swap_old_new):

    if swap_old_new:
        oldmin, newmin = newmin, oldmin
        oldmax, newmax = newmax, oldmax
        value, expected = expected, value

    assert minmax_scaler(value, oldmin, oldmax, newmin, newmax) == expected
    # corner cases:
    assert minmax_scaler(oldmin, oldmin, oldmax, newmin, newmax) == newmin
    assert minmax_scaler(oldmax, oldmin, oldmax, newmin, newmax) == newmax


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
        minmax_scaler(a, np.min(a), np.max(a))  # type: ignore
