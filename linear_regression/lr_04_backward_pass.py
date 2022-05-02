import numpy as np


def loss_gradients(data, weights):
    batch_size = data['X'].shape[0]

    dLdP = - 2 * (data['y'] - data['P'])

    # one matrix
    dPdN = np.ones_like(data['N'])
    dPdB = np.ones_like(data['N'])

    dNdW = np.transpose(data['X'], (1, 0))

    dLdN = dLdP * dPdN

    # gradient 1
    dLdW = np.dot(dNdW, dLdN)

    # gradient 2
    dLdB = (dLdP * dPdB).sum(axis=0)

    gradients = {'W': dLdW, 'B': dLdB}

    return gradients
