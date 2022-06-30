import numpy as np
from fnd_01_derivative import deriv


def matmul_forward(X, W):
    N = np.dot(X, W)
    return N


def matmul_backward(X, W):
    deriv_X = np.transpose(W, (1, 0))
    return deriv_X


X = np.array([2, 3, 5])
W = np.array([4, 9, 3])[:, np.newaxis]

print(matmul_backward(X, W))
