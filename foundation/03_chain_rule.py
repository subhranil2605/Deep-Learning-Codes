import numpy as np
from numpy import ndarray
from typing import Callable, List


# derivative function
def deriv()


def square(x: ndarray) -> ndarray:
    '''
    Evaluates the square of the input value: x**2
    '''
    return x ** 2


def sigmoid(x: ndarray) -> ndarray:
    '''
    Apply the sigmoid function to each element in the input array.
    '''
    return 1 / (1 + np.exp(-x))


# array function
Array_function = Callable[[ndarray], ndarray]

# chain of functions
Chain = List[Array_function]


# Chain Rule
def chain_deriv_2(chain: Chain,
                  input_range: ndarray) -> ndarray:
    '''
    Uses the chain rule to compute the derivative of the composite functions:
    (f2(f1(x)))' = f2'(f1(x)) * f1'(x)
    '''

    assert len(chain) == 2, "This function requires two functions"

    assert input_range.ndim == 1, "1D array of elements"

    f1 = chain[0]
    f2 = chain[1]

    # f1(x)
    f1_of_x = f1(input_range)

    # f1'(x)
