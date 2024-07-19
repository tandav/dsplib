import typing as tp

import numpy as np


class CircularBuffer:
    """
    actually grows "row by row" instead of collection.deque-like fashion
    """

    def __init__(
        self,
        size: tp.Optional[int] = None,
        dtype: type = np.float32,
        index: int = 0,
        buffer: tp.Optional[np.ndarray] = None,
    ):
        self.dtype = dtype
        self.index = index
        if buffer is not None:
            self.buffer = buffer
            if size is not None:
                raise ValueError('size must be None if buffer is passed')
            self.size = buffer.shape[0]
        else:
            if size is None:
                raise ValueError('size must be passed if buffer is None')
            self.buffer = np.zeros(size, dtype=dtype)
            self.size = size

    def append(self, v: float) -> None:
        self.buffer[self.index] = v
        self.index = (self.index + 1) % self.size

    def extend(self, x: np.ndarray) -> None:
        l = x.shape[0]
        if l <= self.size:
            if self.index + l < self.size:
                self.buffer[self.index:self.index + l] = x
            else:
                n_append_end = self.size - self.index
                self.buffer[self.index:] = x[:n_append_end]
                self.buffer[:l - n_append_end] = x[n_append_end:]
        else:
            mod = l % self.size
            q = mod - (self.size - self.index)
            self.buffer[:q] = x[-q:]
            self.buffer[q:] = x[-self.size:-q]  # pylint: disable=invalid-unary-operand-type
        self.index = (self.index + l) % self.size

    def most_recent(self, n: tp.Optional[int] = None) -> np.ndarray:
        if n is None:
            n = self.size
        elif n > self.size:
            raise ValueError(f'n cant be greater than size {n} > {self.size}')
        elif n <= self.index:
            return self.buffer[self.index - n:self.index]
        return np.hstack((self.buffer[-(n - self.index):], self.buffer[:self.index]))
