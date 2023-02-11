import numpy as np
import pytest

from dsplib.window import chunkify
from dsplib.window import make_windows


@pytest.mark.xfail
def test_make_correlation_matrix():
    raise NotImplementedError


def test_chunkify_signal():
    signal = np.arange(20)
    chunked = chunkify(signal, chunk_size=5, n_overlap=2)
    expected = np.array([
        [0,  1,  2,  3,  4],
        [3,  4,  5,  6,  7],
        [6,  7,  8,  9, 10],
        [9, 10, 11, 12, 13],
        [12, 13, 14, 15, 16],
        [15, 16, 17, 18, 19],
    ])
    assert np.array_equal(chunked, expected)


def test_make_windows():
    a = np.arange(7)
    sizes = 3, 4, 5
    windows = make_windows(a, sizes)
    expected = [
        np.array([
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
        ]),
        np.array([
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
        ]),
        np.array([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
        ]),
    ]
    assert all(np.array_equal(w, e) for w, e in zip(windows, expected))

    for window, w_size in zip(windows, sizes, strict=True):
        assert window.shape == (a.shape[0] - max(sizes) + 1, w_size)
