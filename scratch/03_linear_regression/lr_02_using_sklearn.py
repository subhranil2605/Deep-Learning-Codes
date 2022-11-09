import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

plt.style.use('seaborn-whitegrid')

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

data = np.hstack([
    raw_df.values[::2, :],
    raw_df.values[1::2, :2]
])

target = raw_df.values[1::2, 2]

# Data Preparation
s = StandardScaler()
data = s.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# regression
lr = LinearRegression(fit_intercept=True)

lr.fit(X_train, y_train)

preds = lr.predict(X_test)

plt.scatter(preds, y_test)
plt.plot([0, 51], [0, 51])
plt.xlabel("Predicted value")
plt.ylabel("Actual value")
plt.title("Predicted vs. Actual values for\nnLinear Regression model")
plt.xlim([0, 51])
plt.ylim([0, 51])

plt.show()
