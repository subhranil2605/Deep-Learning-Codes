from linear_regression.lr_08_train import train
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
                   batch_size=40,
                   return_losses=True,
                   return_weights=True,
                   seed=180708)

losses = train_info[0]
weights = train_info[1]


import matplotlib.pyplot as plt

plt.plot(list(range(1000)), losses)
plt.show()