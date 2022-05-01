import numpy as np


def matrix_forward(X, W, func):
    N = np.dot(X, W)
    S = func(N)

    L = np.sum(S)

    return S, L


np.random.seed(4645)

X = np.random.randn(3, 3)
W = np.random.randn(3, 2)

print(matrix_forward(X, W, np.sin))