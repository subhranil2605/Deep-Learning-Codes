import numpy as np
from numpy import ndarray
from typing import Dict, Tuple

# weight type
Str_array_dict = Dict[str, ndarray]


def forward_linear_regression(X_batch: ndarray,
                              y_batch: ndarray,
                              weights: Str_array_dict) -> Tuple[ndarray, Weight]:
    '''
    Forward pass for the step-by-step linear regression
    '''

    N = np.dot(X_batch, weights['W'])

    P = N + weights['B']

    loss = np.mean(np.power(y_batch - P), 2)

    forward_info = {
        'X': X_batch,
        'N': N,
        'P': P,
        'y': y_batch
    }

    return loss, forward_info


# Calculating the gradient
def loss_gradients(forward_info: Str_array_dict,
                   weights: Str_array_dict) -> Str_array_dict:
    '''
    Computes the gradients
    '''

    batch_size = forward_info['X'].shape[0]

    dLdP = - 2 * (forward_info['y'] - forward_info['P'])

    dPdN = np.ones_like(forward_info['N'])

    dPdB = np.ones_like(weights['B'])

    dLdN = dLdP * dPdN

    dNdW = np.transpose(forward_info['X'], (1, 0))

    dLdW = np.dot(dNdW, dLdN)

    dLdB = (dLdP * dPdB).sum(axis=0)

    loss_gradient = {
        'W': dLdW,
        'B': dLdB
    }

    return loss_gradient


def init_weights(n_in):
    weights = {
        'W': np.random.randn(n_in, 1),
        'B': np.random.randn(1, 1)
    }

    return weights


def generate_batch(X, y, start: int = 0,
                   batch_size: int = 10):
    '''
    Generate batch from X and y, given a start position
    '''

    if start + batch_size > X.shape[0]:
        batch_size = X.shape[0] - start

    X_batch, y_batch = X[start:start + batch_size], y[start:start + batch_size]

    return X_batch, y_batch


def permute_data(X, y):
    '''
    Permute X and y, using the same permutation, along axis=0
    '''
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


def forward_loss(X, y, weights):
    '''
    Generate predictions and calculate loss for a step-by-step linear regression
    (used mostly during inference).
    '''
    N = np.dot(X, weights['W'])

    P = N + weights['B']

    loss = np.mean(np.power(y - P, 2))

    forward_info = {
        'X': X,
        'N': N,
        'P': P,
        'y': y
    }

    return forward_info, loss


def train(X, y, n_iter=1000, learning_rate=0.01, batch_size=100, return_losses=False,
          return_weights=False, seed=1):
    if seed:
        np.random.seed(seed)

    start = 0

    weights = init_weights(X.shape[1])

    X, y = permute_data(X, y)

    if return_losses:
        losses = []

    for i in range(n_iter):
        if start >= X.shape[0]:
            X, y = permute_data(X, y)
            start = 0

        X_batch, y_batch = generate_batch(X, y, start, batch_size)
        start += batch_size

        forward_info, loss = forward_loss(X_batch, y_batch, weights)

        if return_losses:
            losses.append(loss)

        loss_grads = loss_gradients(forward_info, weights)

        for key in weights.keys():
            weights[key] -= learning_rate * loss_grads[key]

    if return_weights:
        return losses, weights

    return None


