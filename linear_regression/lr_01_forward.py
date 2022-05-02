import numpy as np


def forward_linear_regression(X_batch, y_batch, weights):

    # matrix multiplication of X_batch with W
    N = np.dot(X_batch, weights['W'])

    # Adding bias to get the predicitons
    P = N + weights['B']

    # calculating the loss
    loss = np.mean(np.power(y_batch - P, 2))

    forward_info = {
        'X': X_batch,
        'N': N,
        'P': P,
        'y': y_batch
    }

    return loss, forward_info



    
