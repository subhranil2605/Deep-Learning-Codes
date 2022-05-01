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


# Chain Derivative
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
    df1dx = deriv(f1, input_range)

    # f2'(f1(x))
    df2du = deriv(f2, f1_of_x)

    return df2du * df1dx


def chain_length_2(chain: Chain,
                   x: ndarray) -> ndarray:
    '''
    Evaluates two functions in a row, in a "Chain"
    '''

    assert len(chain) == 2, "Length of input 'chain' should be 2"

    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(x))


plot_range = np.linspace(-3, 3, 1000)

chain_1 = [sigmoid, square]
chain_2 = [square, sigmoid]

# sigmoid(square(x))
y1 = chain_deriv_2(chain_1, plot_range)
y2 = chain_deriv_2(chain_2, plot_range)

plt.subplot(1, 2, 1)
plt.plot(plot_range, chain_length_2(chain_1, plot_range), label="f(x)")
plt.plot(plot_range, y1, label="df/dx")
plt.title("$f(x)=(\sigma(x))^2$")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(plot_range, chain_length_2(chain_2, plot_range), label="f(x)")
plt.plot(plot_range, y2, label="df/dx")
plt.title("$f(x)=\sigma(x^2)$")
plt.legend()

plt.show()
