import numpy as np


class CircularBuffer:
    """
    actually grows "row by row" instead of collection.deque-like fashion
    """

    def __init__(
        self, size: int | None = None,
        dtype: type = np.float32,
        index: int = 0,
        buffer: np.ndarray | None = None,
    ):
        self.dtype = dtype
        self.index = index
        if buffer is not None:
            self.buffer = buffer
            assert size is None, 'size must be None if buffer is passed'
            self.size = buffer.shape[0]
        else:
            self.buffer = np.zeros(size, dtype=dtype)
            self.size = size

    def append(self, v):
        self.buffer[self.index] = v
        self.index = (self.index + 1) % self.size

    def extend(self, x: np.ndarray):
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
            self.buffer[q:] = x[-self.size:-q]
        self.index = (self.index + l) % self.size

    def most_recent(self, n: int):
        if n > self.size:
            raise ValueError('n cant be greater than size')
        if n <= self.index:
            return self.buffer[self.index - n:self.index]
        return np.hstack((self.buffer[-(n - self.index):], self.buffer[:self.index]))
