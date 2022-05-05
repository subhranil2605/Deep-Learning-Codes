import numpy as np

from neural_network_from_scratch.nn_01_sigmoid import sigmoid


def loss_gradients(data, weights):
    dLdP = - (data['y'] - data['P'])

    dPdM2 = np.ones_like(data['M2'])
    dPdB2 = np.ones_like(data['B2'])

    dLdM2 = dLdP * dPdM2

    dLdB2 = (dLdP * dPdB2).sum(axis=0)

    dM2dW2 = np.transpose(data['O1'], (1, 0))

    dLdW2 = np.dot(dM2dW2, dLdP)

    dM2dO1 = np.transpose(data['W2'], (1, 0))

    dLdO1 = np.dot(dLdM2, dM2dO1)

    dO1dN1 = sigmoid(data['N1']) * (1 - sigmoid(data['N1']))

    dLdN1 = dLdO1 * dO1dN1

    dN1dB1 = np.ones_like(weights['B1'])
    dN1dM1 = np.ones_like(weights['M1'])

    dLdB1 = (dLdN1 * dN1dB1).sum(axis=0)

    dLdM1 = dLdN1 * dN1dM1

    dM1dW1 = np.transpose(data['X'], (1, 0))

    dLdW1 = np.dot(dM1dW1, dLdM1)

    return {'W2': dLdW2, 'B2': dLdB2.sum(axis=0), 'W1': dLdW1, 'B1': dLdB1.sum(axis=0)}
