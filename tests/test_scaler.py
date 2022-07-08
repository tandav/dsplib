import pytest
import dsptools


# test negative
# test zero
@pytest.mark.parametrize('value, oldmin, oldmax, newmin, newmax, expected', [
    (50, 0, 100, 0.0, 1.0, 0.5),
    (255, 0, 255, 0.0, 1.0, 1.0),
])
def test_minmax_scaler(value, oldmin, oldmax, newmin, newmax, expected):
    assert dsptools.minmax_scaler(value, oldmin, oldmax, newmin, newmax) == expected