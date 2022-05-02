import numpy as np


def predict(X, weights):
    N = np.dot(X, weights['W'])
    P = N + weights['B']
    return P
