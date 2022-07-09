import numpy as np
import pytest

from dsptool.chunks import chunkify_signal


@pytest.mark.xfail
def test_make_correlation_matrix():
    raise NotImplementedError


def test_chunkify_signal():
    signal = np.arange(20)
    chunked = chunkify_signal(signal, chunk_size=5, n_overlap=2)
    expected = np.array([
        [0,  1,  2,  3,  4],
        [3,  4,  5,  6,  7],
        [6,  7,  8,  9, 10],
        [9, 10, 11, 12, 13],
        [12, 13, 14, 15, 16],
        [15, 16, 17, 18, 19],
    ])
    assert np.array_equal(chunked, expected)
