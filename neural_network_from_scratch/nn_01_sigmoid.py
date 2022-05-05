import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(- 1.0 * x))


# x = np.linspace(-5, 5, 100)
# plt.plot(x, sigmoid(x))
#
# plt.show()