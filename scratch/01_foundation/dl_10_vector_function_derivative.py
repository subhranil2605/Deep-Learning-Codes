from numpy import ndarray
import numpy as np
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


def matrix_function_backward(X: ndarray,
                             W: ndarray,
                             sigma: Array_function) -> ndarray:
    '''
    Computes the forward pass of a function involving matrix multiplication
    with extra function
    '''

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # backward calculation
    dSdN = deriv(sigma, N)

    # dNdX
    dNdX = np.transpose(W, (1, 0))

    # multiply them together: since dNdX is 1x1 here, order doesn't matter
    return np.dot(dSdN, dNdX)
