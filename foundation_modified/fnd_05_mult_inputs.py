from fnd_01_derivative import deriv
import numpy as np


def multiple_inputs_add(x, y, func):
    a = x + y

    return func(a)


def multiple_inputs_deriv(x, y, func):
    a = x + y

    # ds/da
    dsda = deriv(func, a)

    dadx = 1
    dady = 1

    return dsda * dadx, dsda * dady


X = np.random.randn(100, 100)
Y = np.random.randn(100, 100)

r = multiple_inputs_add(X, Y, np.sin)

dfdx = multiple_inputs_deriv(X, Y, np.sin)[0]
dfdy = multiple_inputs_deriv(X, Y, np.sin)[1]

