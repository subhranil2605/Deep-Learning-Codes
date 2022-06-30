from numpy import ndarray
import numpy as np
from typing import Callable

Array_function = Callable[[ndarray], ndarray]


def matrix_forward_extra(X: ndarray,
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

    return S
