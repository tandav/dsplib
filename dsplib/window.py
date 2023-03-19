import typing as tp

import numpy as np


def chunkify(signal: np.ndarray, chunk_size: int = 256, n_overlap: int = 32) -> np.ndarray:
    """todo: see
        - https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
        - https://github.com/pydata/bottleneck
    """
    step = chunk_size - n_overlap
    shape = ((signal.shape[-1] - n_overlap) // step, chunk_size)
    strides = (step * signal.strides[-1], signal.strides[-1])
    result = np.lib.stride_tricks.as_strided(
        signal, shape=shape, strides=strides,
    )
    return result


def make_windows(a: np.ndarray, sizes: tp.Tuple[int, ...]) -> tp.List[np.ndarray]:
    """TODO: just create window for max size and then use slices [:, -size:] to make smaller windows
        - but it can be slower because slicing returns a copy instead of view
        - or not: https://scipy-cookbook.readthedocs.io/items/ViewsVsCopies.html#Slice-views
    """
    if any(size < 1 for size in sizes):
        raise ValueError('window size should be >= 1')
    windows = []

    # # multiple sliding_window_view
    # max_size = max(sizes)
    # for w_size in sizes:
    #     windows.append(
    #         np.lib.stride_tricks.sliding_window_view(
    #             a, window_shape=w_size,
    #         )[max_size - w_size:],
    #     )

    # single sliding_window_view and slices
    _sizes = sorted(sizes)
    max_window = np.lib.stride_tricks.sliding_window_view(
        a, window_shape=sizes[-1],
    )
    for w_size in _sizes:
        windows.append(max_window[:, -w_size:])

    return windows
