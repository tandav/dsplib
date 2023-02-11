import numpy as np
import pytest

from dsplib.circular_buffer import CircularBuffer


def test_append():
    b = CircularBuffer(size=2)
    b.append(1)
    assert np.array_equal(b.buffer, np.array([1, 0]))


def test_multiple_append():
    """test that index is working correctly"""
    b = CircularBuffer(size=2)
    b.append(1)
    b.append(2)
    assert np.array_equal(b.buffer, np.array([1, 2]))
    b.append(3)
    assert np.array_equal(b.buffer, np.array([3, 2]))


@pytest.mark.parametrize(
    'index, buffer, data, expected', [
        (0, np.zeros(7), np.arange(4), np.array([0, 1, 2, 3, 0, 0, 0])),
        (4, np.array([0, 1, 2, 3, 0, 0, 0, 0, 0, 0]), np.arange(10, 18), np.array([16, 17, 2, 3, 10, 11, 12, 13, 14, 15])),
        (4, np.arange(7), np.arange(10, 29), np.array([27, 28, 22, 23, 24, 25, 26])),
    ],
)
def test_extend(data, expected, index, buffer):
    b = CircularBuffer(index=index, buffer=buffer)
    b.extend(data)
    assert np.array_equal(b.buffer, expected)


@pytest.mark.parametrize(
    'index, buffer, data, expected', [
        (0, np.zeros(7), [np.arange(4), np.arange(2)], np.array([0, 1, 2, 3, 0, 1, 0])),
        (
            4,
            np.array([0, 1, 2, 3, 0, 0, 0, 0, 0, 0]),
            [
                np.arange(10, 18),
                np.array([33, 34]),
            ],
            np.array([16, 17, 33, 34, 10, 11, 12, 13, 14, 15]),
        ),
        (
            4,
            np.arange(7),
            [
                np.arange(10, 29),
                np.array([0, 1]),
            ],
            np.array([27, 28, 0, 1, 24, 25, 26]),
        ),
    ],
)
def test_multiple_extend(index, buffer, data, expected):
    b = CircularBuffer(index=index, buffer=buffer)
    for _data in data:
        b.extend(_data)
    assert np.array_equal(b.buffer, expected)


@pytest.mark.parametrize(
    'index, buffer, n, expected', [
        (2, np.array([0, 1, 2, 3, 4]), 2, np.array([0, 1])),
        (2, np.array([0, 1, 2, 3, 4]), 3, np.array([4, 0, 1])),
        (2, np.array([0, 1, 2, 3, 4]), 5, np.array([2, 3, 4, 0, 1])),
    ],
)
def test_most_recent(index, buffer, n, expected):
    b = CircularBuffer(index=index, buffer=buffer)
    assert np.array_equal(b.most_recent(n), expected)
