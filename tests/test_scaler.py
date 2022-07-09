import pytest
from dsptool.core import minmax_scaler
from dsptool.core import minmax_scaler_array

@pytest.mark.parametrize('value, oldmin, oldmax, newmin, newmax', [
    # oldmin, oldmax
    (0, 0, 0, 0, 1),
    (0, 0, -1, 0, 1),
    (0, 10, 10, 0, 1),
    (0, 10, 9, 0, 1),

    # newmin, newmax
    (50, 0, 100, 0, 0),
    (50, 0, 100, 0, -1),
    (50, 0, 100, 10, 10),
    (50, 0, 100, 10, 9),
])
def test_min_less_than_max(value, oldmin, oldmax, newmin, newmax):
    with pytest.raises(ValueError):
        assert minmax_scaler(value, oldmin, oldmax, newmin, newmax)


@pytest.mark.parametrize('value, oldmin, oldmax, newmin, newmax', [
    (101, 0, 100, 0.0, 1.0),
    (-1, 0, 100, 0.0, 1.0),
])
def test_value_inside_bounds(value, oldmin, oldmax, newmin, newmax):
    with pytest.raises(ValueError):
        assert minmax_scaler(value, oldmin, oldmax, newmin, newmax)


@pytest.mark.parametrize('swap_old_new', [False, True])
@pytest.mark.parametrize('value, oldmin, oldmax, newmin, newmax, expected', [
    pytest.param(-18, -90, -10, 0, 1, 0.9, id='--'),
    pytest.param(-10, -100, 0, 0, 1, 0.9, id='-0'),
    pytest.param(80, -10, 90, 0, 1, 0.9, id='-+'),
    pytest.param(90, 0, 100, 0, 1, 0.9, id='0+'),
    pytest.param(82, 10, 90, 0, 1, 0.9, id='++'),
])
def test_minmax_scaler(value, oldmin, oldmax, newmin, newmax, expected, swap_old_new):

    if swap_old_new:
        oldmin, newmin = newmin, oldmin
        oldmax, newmax = newmax, oldmax
        value, expected = expected, value

    assert minmax_scaler(value, oldmin, oldmax, newmin, newmax) == expected
    # corner cases:
    assert minmax_scaler(oldmin, oldmin, oldmax, newmin, newmax) == newmin
    assert minmax_scaler(oldmax, oldmin, oldmax, newmin, newmax) == newmax


@pytest.mark.xfail
def minmax_scaler_array():
    raise NotImplementedError
