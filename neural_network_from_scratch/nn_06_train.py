import numpy as np

from linear_regression.lr_06_permute_data import permutate_data
from linear_regression.lr_07_generate_batch import generate_batch
from neural_network_from_scratch.nn_02_init_params import init_weights
from neural_network_from_scratch.nn_03_forward_loss import forward_loss
from neural_network_from_scratch.nn_04_loss_gradients import loss_gradients
from neural_network_from_scratch.nn_05_predict import predict


def train(X_train, y_train, X_test, y_test, n_iter,
          test_every=1000, learning_rate=0.01,
          hidden_size=13, batch_size=100,
          return_losses=False, return_weights=False,
          return_scores=False, seed=1):
    if seed:
        np.random.seed(seed)

    start = 0

    weights = init_weights(X_train.shape[1], hidden_size=hidden_size)

    # permute data
    X_train, y_train = permutate_data(X_train, y_train)

    losses = []
    val_scores = []

    for i in range(n_iter):

        if start >= X_train.shape[0]:
            X_train, y_train = permutate_data(X_train, y_train)
            start = 0

        X_batch, y_batch = generate_batch(X_train, y_train, start, batch_size)
        start += batch_size

        forward_info, loss = forward_loss(X_batch, y_batch, weights)

        if return_losses:
            losses.append(loss)

        loss_grads = loss_gradients(forward_info, weights)

        for key in weights.keys():
            weights[key] -= learning_rate * loss_grads[key]

        # if return_scores:
        #     if i % test_every == 0 and i != 0:
        #         preds = predict(X_test, weights)
        #         val_scores.append()

    if return_weights:
        return losses, weights, val_scores

    return None

