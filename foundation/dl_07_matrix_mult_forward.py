from numpy import ndarray
import numpy as np


def matmul_forward(X: ndarray,
                   W: ndarray) -> ndarray:
    '''
    Computes the forward pass of a matrix multiplication
    '''

    assert X.shape[1] == W.shape[0], \
        '''
        For matrix multiplication, the number of columns in the first array should
        match the number of rows in the second;
        '''

    # matrix multiplication
    N = np.dot(X, W)

    return N
