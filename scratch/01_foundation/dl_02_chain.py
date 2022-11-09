import numpy as np
from numpy import ndarray
from typing import Callable, List

# A function takes in a ndarray as argument and produces an ndarray
Array_function = Callable[[ndarray], ndarray]

# A chain is a list of functions
Chain = List[Array_function]


def chain_length_2(chain: Chain,
                   x: ndarray) -> ndarray:
    '''
    Evaluates two functions in a row, in a "Chain"
    '''

    assert len(chain) == 2, "Length of input 'chain' should be 2"

    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(x))


# two functions
square = lambda x: x ** 2
quad = lambda x: x ** 3

chain_functions = [square, quad]

# values
x = np.arange(4, 6)

# showing the result
print(chain_length_2(chain_functions, x))
