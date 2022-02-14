"""
Implements common activation functions for feed-forward neural networks.
"""

import numpy as np


def identity(x):
    return x

def identity_gradient(x):
    return 1.0

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_gradient(x):
    s = sigmoid(x)
    return  s * (1 - s)

def tanh(x):
    return np.tanh(x)  # 2 * sigmoid(2*x) - 1.0

def tanh_gradient(x):
    return 1 - np.square(np.tanh(x))

def ReLU(x):
    return x *  (x > 0)

def ReLU_gradient(x):
    return 1. * (x > 0)

def softmax(x):
    m_x = np.max(x, axis=1).reshape(-1, 1)  
    e_x = np.exp(x - m_x)  # subtract max for numerical stability
    return e_x / np.sum(e_x, axis=1).reshape(-1, 1)

def softmax_gradient(x):
    return np.ones(x.shape)  # cross_entropy softmax gradient shortcut


activations = {
    'identity': (identity, identity_gradient),
    'sigmoid': (sigmoid, sigmoid_gradient),
    'tanh': (tanh, tanh_gradient),
    'ReLU': (ReLU, ReLU_gradient),
    'softmax': (softmax, softmax_gradient)
}