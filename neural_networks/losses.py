"""
Implements common loss functions for feed-forward neural networks.
"""

import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def mean_squared_error_gradient(y_true, y_pred):
    return y_pred - y_true

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_absolute_error_gradient(y_true, y_pred):
    return np.sign(y_pred - y_true)

def cross_entropy(y_true, y_pred, epsilon=1e-6):
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  # for numerical stability
    return -np.mean(np.multiply(y_true, np.log(y_pred)))

def cross_entropy_gradient(y_true, y_pred, epsilon=1e-6):
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  # for numerical stability
    # return - y_true / y_pred
    return y_pred -y_true

losses = {
    'mean_squared_error': (mean_squared_error, mean_squared_error_gradient),
    'mean_absolute_error': (mean_absolute_error, mean_absolute_error_gradient),
    'cross_entropy': (cross_entropy, cross_entropy_gradient)
}