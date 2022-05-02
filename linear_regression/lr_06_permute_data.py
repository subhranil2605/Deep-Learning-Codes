import numpy as np


def permutate_data(X, y):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]

