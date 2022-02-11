import numpy as np
from numpy.linalg import norm

WEIGHT, BIAS = 0, 1  # Enumerator for indexing optimizer parameters


def clip_gradients(gradients, clip_ratio=1.0, norm_type=2):

    def normalize(gradient):
        return (clip_ratio * gradient / norm(gradient, ord=norm_type))
    
    return [(normalize(d_W), normalize(d_b)) for (d_W, d_b) in gradients]


class Optimizer():
    '''
    Implements optimization algorithms to train a neural network.
    '''

    def __init__(self, model, algorithm='sgd', **kwargs):
        self.model = model
        self.algorithm = algorithm

        # Variables
        if self.algorithm in ['momentum']:
            self.v_t = [[np.zeros(l.W.shape), np.zeros(l.b.shape)]  # weights, biases
                        for l in self.model.layers]
        self.__dict__.update(kwargs)

    
    def backpropagate(self, y_true, y_pred, loss):
        '''Calculates and returns gradients for each layer.'''
        gradients, batch_size = [], y_pred.shape[0]
        _, loss_grad = loss

        layers = self.model.layers.copy()

        for l, layer in enumerate(reversed(layers)):
            
            if l == 0:  # output layer
                d_L = loss_grad(y_true, y_pred)
                delta = np.multiply(d_L, layer.gradient())
                d_W = np.matmul(layer.input.T, delta) / batch_size
                d_b = np.mean(delta, axis=0) if layer.use_bias else np.zeros(layer.b.shape)
            else:  # input and hidden layers
                delta = np.multiply(np.matmul(delta, next_layer.W.T), layer.gradient())
                d_W = np.matmul(layer.input.T, delta) / batch_size
                d_b = np.mean(delta, axis=0) if layer.use_bias else np.zeros(layer.b.shape)

            next_layer = layer
            gradients.append((d_W, d_b))
        
        return gradients[::-1]
    
    def optimize(self, gradients, learning_rate=4e-3, **kwargs):

        def sgd(gradients, learning_rate):
            for l, (d_W, d_b) in enumerate(gradients):
                self.model.layers[l].W -= learning_rate * d_W
                self.model.layers[l].b -= learning_rate * d_b
        
        def momentum(gradients, learning_rate):
            for l, grad in enumerate(gradients):
                for param in [WEIGHT, BIAS]:
                    self.v_t[l][param] = ((self.momentum * self.v_t[l][param]) + (learning_rate * grad[param]))
                self.model.layers[l].W -= self.v_t[l][WEIGHT]
                self.model.layers[l].b -= self.v_t[l][BIAS]
        
        self.__dict__.update(kwargs)

        optimizer = {
            'sgd': sgd,
            'momentum': momentum,
        }
        optimizer[self.algorithm](gradients, learning_rate, **kwargs)