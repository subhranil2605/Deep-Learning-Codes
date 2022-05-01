import numpy as np
from typing import Callable

Array_function = Callable[[np.ndarray], np.ndarray]


# derivative function
def deriv(func: Array_function,
          input_: np.ndarray,
          delta: float = 0.001) -> np.ndarray:
    '''
    Evaluates the derivative of a function "func" at every element
    '''
    return (func(input_ + delta) - func(input_ - delta)) / (delta * 2)


# derivative of the multiple input function
def multiple_input_derivative(x: np.ndarray,
                              y: np.ndarray,
                              sigma: Array_function) -> float:
    '''
    Computes the derivative of this simple function with respect to both inputs
    '''

    # compute the "forward pass"
    a = x + y

    # compute the derivatives
    dsda = deriv(sigma, a)

    dadx = dady = 1

    return dsda * dadx, dsda * dady
