import numpy as np


def init_params(n_in):
    weights = {
        'W': np.random.randn(n_in, 1),
        'B': np.random.randn(1, 1)
    }
    return weights


