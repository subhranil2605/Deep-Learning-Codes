import numpy as np
import matplotlib.pyplot as plt

def chain_len_2(chain, x):
    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(x))

sigmoid = lambda x: 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 1000)

chain_1 = [np.sin, sigmoid]
chain_2 = [sigmoid, np.sin]

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x))

plt.subplot(2, 2, 2)
plt.plot(x, np.sin(x))

plt.subplot(2, 2, 3)
plt.plot(x, chain_len_2(chain_1, x))
plt.title("$\sigma(\sin(x))$")


plt.subplot(2, 2, 4)
plt.plot(x, chain_len_2(chain_2, x))


plt.show()
