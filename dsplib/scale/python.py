def _within_bounds(
    value: float,
    min_: float,
    max_: float,
) -> bool:
    if min_ == max_:
        return value == min_
    min_, max_ = min(min_, max_), max(min_, max_)
    return min_ <= value <= max_


def minmax_scaler(
    value: float,
    oldmin: float,
    oldmax: float,
    newmin: float = 0.0,
    newmax: float = 1.0,
) -> float:
    if not _within_bounds(value, oldmin, oldmax):
        raise ValueError('value should be oldmin <= value <= oldmax')

    if oldmin == oldmax:
        if newmin != newmax:
            raise ValueError('oldmin == oldmax, so newmin == newmax must be true')
        return newmin

    return (value - oldmin) * (newmax - newmin) / (oldmax - oldmin) + newmin
