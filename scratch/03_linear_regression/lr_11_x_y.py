import sklearn.metrics

from linear_regression.lr_08_train import train
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from linear_regression.lr_10_predict import predict
import matplotlib.pyplot as plt


def sign(X, y):
    if X > y:
        return 1
    else:
        return -1


def data_gen_x(n):
    a = []
    for i in range(n):
        X = np.random.randint(0, 400)
        y = np.random.randint(0, 400)
        a.append([X, y])

    return np.array(a)


def data_gen_y(X):
    a = []
    for i in range(X.shape[0]):
        a.append(sign(X[i, 0], X[i, 1]))
    return np.array(a).reshape(X.shape[0], 1)


data = data_gen_x(1000)
target = data_gen_y(data)

# data prep
s = StandardScaler()
data = s.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)

y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

print(X_train.shape)

# manually training
train_info = train(X_train, y_train,
                   n_iter=1000,
                   learning_rate=0.001,
                   batch_size=5,
                   return_losses=True,
                   return_weights=True,
                   seed=180708)

losses = train_info[0]
weights = train_info[1]

preds = predict(X_test, weights)


print("slope: ", weights['W'][0]/weights['W'][1])
print("bias: ", weights['B'])