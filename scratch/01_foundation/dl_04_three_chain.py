import numpy as np
from numpy import ndarray
from typing import Callable, List
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


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


def leaky_relu(x: ndarray) -> ndarray:
    alpha = 0.1
    return np.maximum(alpha * x, x)


# array function
Array_function = Callable[[ndarray], ndarray]

# chain of functions
Chain = List[Array_function]


# derivative function
def deriv(func: Array_function,
          input_: np.ndarray,
          delta: float = 0.001) -> np.ndarray:
    '''
    Evaluates the derivative of a function "func" at every element
    '''
    return (func(input_ + delta) - func(input_ - delta)) / (delta * 2)


# Chain Rule
def chain_deriv_3(chain: Chain,
                  input_range: ndarray) -> ndarray:
    '''
    Uses the chain rule to compute the derivative of the composite functions:
    (f3(f2(f1(x))))' = f3'(f2(f1(x))) * f2'(f1(x)) * f1'(x)
    '''

    assert len(chain) == 3, "This function requires two functions"

    assert input_range.ndim == 1, "1D array of elements"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    # f1(x)
    f1_x = f1(input_range)

    # f2(f1(x))
    f2_f1_x = f2(f1_x)

    # df3du
    df3du = deriv(f3, f2_f1_x)

    # df2du
    df2du = deriv(f2, f1_x)

    # df1dx
    df1dx = deriv(f1, input_range)

    return df1dx * df2du * df3du


def chain_length_3(chain: Chain,
                   x: ndarray) -> ndarray:
    '''
    Evaluates three functions in a row, in a "Chain"
    '''

    assert len(chain) == 3, "Length of input 'chain' should be 3"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    return f3(f2(f1(x)))


plot_range = np.linspace(-3, 3, 1000)

chain = [leaky_relu, sigmoid, square]


# sigmoid(square(x))
y = chain_deriv_3(chain, plot_range)


plt.plot(plot_range, chain_length_3(chain, plot_range), label="f(x)")
plt.plot(plot_range, y, label="df/dx")
plt.title("$f(x)=LeRU(\sigma(x))^2)$")
plt.legend()



plt.show()
