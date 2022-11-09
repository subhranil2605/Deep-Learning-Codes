def generate_batch(X, y, start=0, batch_size=10):
    if start + batch_size > X.shape[0]:
        batch_size = X.shape[0] - start

    X_batch = X[start:start + batch_size]
    y_batch = y[start:start + batch_size]

    return X_batch, y_batch
