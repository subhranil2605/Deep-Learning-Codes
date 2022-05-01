from fnd_01_derivative import *
import numpy as np
import matplotlib.pyplot as plt


def chain_len_3(chain, input_range):
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    return f3(f2(f1(input_range)))


def chain_len_3_deriv(chain, input_range):
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    # let say u = f2(f1(x))
    # and v = f1(x)

    v = f1(input_range)
    u = f2(v)

    # df1/dx
    df1dx = deriv(f1, f1(input_range))

    # df2/dv
    df2dv = deriv(f2, v)

    # df3/du
    df3du = deriv(f3, u)

    # the total derivative
    dydx = df3du * df2dv * df1dx

    return dydx


X = np.linspace(-3, 3, 100)

chain = [np.sin, np.cos, np.tan]

print(X)
f = chain_len_3(chain, X)
df = chain_len_3_deriv(chain, X)

plt.subplot(2, 2, 1)
plt.plot(X, f)
plt.title("$tan(cos(sin(x)))$")

plt.subplot(2, 2, 2)
plt.plot(X, df)
plt.title("$d{tan(cos(sin(x)))}/dx$")


j = np.tan(np.cos(np.sin(X)))
plt.subplot(2, 2, 3)
plt.plot(X, j)
plt.title("$d{tan(cos(sin(x)))}/dx$")


plt.show()
