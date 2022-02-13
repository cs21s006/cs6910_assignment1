import numpy as np


def make_batches(X, y, batch_size=32):
    l, b = X.shape[0], batch_size
    for i in range(0, l, b):
        j = min(i+b, l)
        yield (X[i:j, :], y[i:j, :])

def preprocess_data(X, y):
    X_norm = X.reshape(-1, 28 * 28) / 255.0
    y_one_hot = np.squeeze(np.eye(10)[y]).reshape(-1, 10)
    return X_norm, y_one_hot