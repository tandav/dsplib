try:
    import numpy as np  # noqa
except ImportError:
    from dsplib.scale.python import minmax_scaler
else:
    from dsplib.scale.numpy import minmax_scaler  # noqa
