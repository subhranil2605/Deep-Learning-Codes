import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from linear_regression.lr_10_predict import predict
from linear_regression.lr_08_train import train

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# data prep
s = StandardScaler()
data = s.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)


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


# most important feautres vs target and predictions
a = np.repeat(X_test[:, :-1].mean(axis=0, keepdims=True), 40, axis=0)
b = np.linspace(-1.5, 3.5, 40).reshape(40, 1)

test_feature = np.concatenate([a, b], axis=1)
test_preds = predict(test_feature, weights)[:, 0]


plt.scatter(X_test[:, 12], y_test)
plt.plot(np.array(test_feature[:, -1]), test_preds, linewidth=2, c="orange")
plt.ylim([6, 51])
plt.show()