import numpy as np
from numpy import ndarray
from typing import Callable

Array_function = Callable[[ndarray], ndarray]


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
