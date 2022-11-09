import numpy as np
from typing import Callable

Array_function = Callable[[np.ndarray], np.ndarray]


def multiple_inputs_add(x: np.ndarray,
                        y: np.ndarray,
                        sigma: Array_function) -> float:
    '''
    Function with multiple inputs and addition, forward pass.
    '''

    assert x.shape == y.shape

    a = x + y
    return sigma(a)


x = np.array([4, 5, 6])
y = np.array([1, 2, 3])

print(multiple_inputs_add(x, y, lambda m: 1 / (1 + np.exp(-m))))

