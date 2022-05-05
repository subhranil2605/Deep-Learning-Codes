import numpy as np


def init_weights(input_size, hidden_size):
    weights = {
        'W1': np.random.randn(input_size, hidden_size),
        'B1': np.random.randn(1, hidden_size),
        'W2': np.random.randn(hidden_size, 1),
        'B2': np.random.randn(1, 1),
    }
    return weights
