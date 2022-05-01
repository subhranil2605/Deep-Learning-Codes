from foundation.main import DeepLearningMath
import numpy as np

np.random.seed(565)

X = np.random.randn(3, 3)


def square(num):
    return num ** 2


print(X)
D = DeepLearningMath()

print(D.deriv(square, X))
