import sklearn.metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from linear_regression.lr_08_train import train
from linear_regression.lr_10_predict import predict

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

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

# prediction of the model
preds = predict(X_test, weights)
print(f"Mean absolute error: {round(sklearn.metrics.mean_absolute_error(y_test, preds), 4)}")
print(f"Root Mean squared error: {round(np.power(sklearn.metrics.mean_squared_error(y_test, preds), .5), 4)}")

plt.subplot(2, 1, 1)
plt.plot(list(range(1000)), losses)

plt.subplot(2, 1, 2)
plt.scatter(preds, y_test)
plt.plot([0, 51], [0, 51], '-k')
plt.xlim([0, 51])
plt.ylim([0, 51])

plt.show()
