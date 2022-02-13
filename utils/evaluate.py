import numpy as np

def accuracy(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_pred == y_true)

def gradient_sum(gradients):
    return np.sum([np.sum(np.abs(d_W) + np.abs(d_b))
                   for (d_W, d_b) in gradients])