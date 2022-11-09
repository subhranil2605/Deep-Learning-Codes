import numpy as np

from neural_network_from_scratch.nn_01_sigmoid import sigmoid


def predict(X, weights):
    M1 = np.dot(X, weights['W1'])
    N1 = M1 + weights['B1']
    O1 = sigmoid(N1)
    M2 = np.dot(O1, weights['W2'])
    P = M2 + weights['B2']

    return P
