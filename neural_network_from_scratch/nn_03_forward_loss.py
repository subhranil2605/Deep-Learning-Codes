import numpy as np

from neural_network_from_scratch.nn_01_sigmoid import sigmoid


def forward_loss(X, y, weights):
    M1 = np.dot(X, weights['W1'])

    N1 = M1 + weights['B1']

    O1 = sigmoid(N1)

    M2 = np.dot(O1, weights['W2'])

    P = M2 + weights['B2']

    loss = np.mean(np.power(y - P, 2))

    forward_info = {
        'X': X,
        'M1': M1,
        'N1': N1,
        'O1': O1,
        'M2': M2,
        'P': P,
        'y': y,
    }

    return forward_info, loss
