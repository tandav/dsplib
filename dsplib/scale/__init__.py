from dsplib.scale.python import minmax_scaler  # noqa

try:
    import numpy as np  # noqa
except ImportError:
    pass
else:
    from dsplib.scale.numpy import minmax_scaler_array  # noqa pylint: disable=ungrouped-imports
