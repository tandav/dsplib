import numpy as np


def chunkify_signal(signal: np.ndarray, chunk_size: int = 256, n_overlap: int = 32) -> np.ndarray:
    step = chunk_size - n_overlap
    shape = ((signal.shape[-1] - n_overlap) // step, chunk_size)
    strides = (step * signal.strides[-1], signal.strides[-1])
    result = np.lib.stride_tricks.as_strided(
        signal, shape=shape, strides=strides,
    )
    return result
