import numpy as np
from fnd_01_derivative import *
import matplotlib.pyplot as plt

def chain_len_2_deriv(chain, x):
    f1 = chain[0]
    f2 = chain[1]

    # considering f1(x) to be another variable callled u
    u = f1(x)

    # derivative of f2 with respect to u
    df2du = deriv(f2, u)

    # derivative of u / (f1(x)) with respect to x
    dudx = deriv(f1, x)

    # finally multiply them to get the total derivative
    dydx = df2du * dudx

    return dydx


chain = [np.sin, np.cos]

x = np.linspace(-10, 10, 1000)


plt.plot(x, np.cos(np.sin(x)))
plt.plot(x, chain_len_2_deriv(chain, x))

plt.show()


