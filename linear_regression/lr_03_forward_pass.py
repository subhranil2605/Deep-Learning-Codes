import numpy as np


def forward_linear_regression(X_batch, y_batch, weights):
    W = weights['W']
    B = weights['B']

    N = np.dot(X_batch, W)
    P = N + B

    loss = np.mean(np.power(y_batch - P, 2))
    
    data = {
        'X': X_batch,
        'N': N,
        'P': P,
        'y': y_batch
    }

    return data
