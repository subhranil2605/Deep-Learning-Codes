import numpy as np
from numpy import ndarray
from typing import Callable

Array_function = Callable[[ndarray], ndarray]


# derivative function
def deriv(func: Callable[[np.ndarray], np.ndarray],
          input_: np.ndarray,
          delta: float = 0.001) -> np.ndarray:
    '''
    Evaluates the derivative of a function "func" at every element
    '''
    return (func(input_ + delta) - func(input_ - delta)) / (delta * 2)


def matrix_funtion_forward_sum(X: ndarray,
                               W: ndarray,
                               sigma: Array_function) -> float:
    '''
    Computing the result of the forward pass of this function with
    input ndarrays X and W and function sigma.
    '''

    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # sum of all the elements
    L = np.sum(S)

    return L


def matrix_function_backward_sum(X: ndarray,
                                 W: ndarray,
                                 sigma: Array_function) -> ndarray:
    '''
    Compute derivative of matrix function with a sum with respect to the
    first matrix input
    '''

    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output the function
    S = sigma(N)

    # sum of all elements
    L = np.sum(S)

    dLdS = np.ones_like(S)

    # dSdN
    dSdN = deriv(sigma, N)

    # dLdN
    dLdN = dLdS * dSdN

    # dNdX
    dNdX = np.transpose(W, (1, 0))

    # dLdX
    dLdX = np.dot(dSdN, dNdX)

    return dLdX


def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))


np.random.seed(190204)
X = np.random.randn(3, 3)
W = np.random.randn(3, 2)

print("X:\n", X)

print("L:")
print(round(matrix_funtion_forward_sum(X, W, sigmoid), 4))


print()

print("dLdX:")
print(matrix_function_backward_sum(X, W, sigmoid))