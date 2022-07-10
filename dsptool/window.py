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


def make_windows(a: np.ndarray, sizes: tuple[int, ...]) -> list[np.ndarray]:
    if any(size < 1 for size in sizes):
        raise ValueError('window size should be >= 1')
    max_size = max(sizes)
    windows = []
    for w_size in sizes:
        windows.append(
            np.lib.stride_tricks.sliding_window_view(
                a, window_shape=w_size,
            )[max_size - w_size:],
        )
    return windows
