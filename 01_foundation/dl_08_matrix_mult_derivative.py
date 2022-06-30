from numpy import ndarray
import numpy as np


def matmul_backward_first(X: ndarray,
                          W: ndarray) -> ndarray:
    '''
    Computes the backward pass of a matrix multiplication with respect
    to the first element
    '''

    # backward pass
    dNdX = np.transpose(W, (1, 0))

    return dNdX
