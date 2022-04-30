from typing import Callable
import numpy as np


def deriv(func: Callable[[np.ndarray], np.ndarray],
          input_: np.ndarray,
          delta: float = 0.001) -> np.ndarray:
    '''
    Evaluates the derivative of a function "func" at every element
    '''
    return (func(input_ + delta) - func(input_ - delta)) / (delta * 2)
