import numpy as np

from linear_regression.lr_01_forward import forward_linear_regression
from linear_regression.lr_04_backward_pass import loss_gradients
from linear_regression.lr_05_init_params import init_params
from linear_regression.lr_06_permute_data import permutate_data
from linear_regression.lr_07_generate_batch import generate_batch


def train(X, y, n_iter=1000,
          learning_rate=0.001,
          batch_size=100,
          return_losses=False,
          return_weights=False,
          seed=1):
    if seed:
        np.random.seed(seed)

    # start
    start = 0

    # column length
    weights = init_params(X.shape[1])

    # permutate data
    X, y = permutate_data(X, y)

    losses = []

    for i in range(n_iter):

        # generate batch
        if start >= X.shape[0]:
            X, y = permutate_data(X, y)
            start = 0

        X_batch, y_batch = generate_batch(X, y, start, batch_size)
        start += batch_size

        forward_info, loss = forward_linear_regression(X_batch, y_batch, weights)

        if return_losses:
            losses.append(loss)

        loss_grads = loss_gradients(forward_info, weights)

        for key in weights.keys():
            weights[key] -= learning_rate * loss_grads[key]

    if return_weights:
        return losses, weights

    return None
