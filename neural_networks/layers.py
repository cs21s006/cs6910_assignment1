import numpy as np

from . import activations

np.random.seed(42)

def init_weights(input_dim, output_dim, init_method):
    if init_method ==  'Xavier_normal':
        n = input_dim + output_dim
        return np.random.normal(scale=np.sqrt(2/n), size=(input_dim, output_dim))
    elif init_method == 'Xavier_uniform':
        n = input_dim + output_dim
        return np.random.uniform(low=-np.sqrt(6/n), high=np.sqrt(6/n),
                                    size=(input_dim, output_dim))
    elif init_method == 'He_normal':
        return np.random.normal(scale=np.sqrt(2/input_dim), size=(input_dim, output_dim))
    elif init_method == 'He_uniform':
        return np.random.uniform(low=-np.sqrt(6/input_dim), high=np.sqrt(6/input_dim),
                                    size=(input_dim, output_dim))
    else :
        return np.random.randn(input_dim, output_dim)


class Layer():
    '''
    Implements a feed-forward layer for a neural network.

    Input: (batch_size, dimension)
    Output: (batch_size, num_neurons)
    '''
    def __init__(self, input_dim: int, num_neurons: int, activation: str='ReLU',
                 use_bias: bool=True, init_method: str='Xavier_normal'):
        self.W = init_weights(input_dim, num_neurons, init_method)
        self.b = np.zeros(num_neurons)
        self.use_bias = use_bias  # used by optimizer
        self.f, self.d_f = activations[activation]
    
    def forward(self, x):
        self.input = x
        return self.f(np.matmul(x, self.W) + self.b)
    
    def gradient(self):
        g = self.d_f(np.matmul(self.input, self.W) + self.b)
        return g
