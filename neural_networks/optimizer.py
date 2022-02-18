import numpy as np
from numpy.linalg import norm

WEIGHT, BIAS = 0, 1  # Enumerator for indexing optimizer parameters


def clip_gradients(gradients, clip_ratio=1.0, norm_type=2, epsilon=1e-8):

    def normalize(gradient):
        return (clip_ratio * gradient / (norm(gradient, ord=norm_type) + epsilon))
    
    return [(normalize(d_W), normalize(d_b)) for (d_W, d_b) in gradients]


class Optimizer():
    '''
    Implements optimization algorithms to train a neural network.
    '''

    def __init__(self, model, algorithm='sgd', **kwargs):
        self.model = model
        self.algorithm = algorithm

        # Variables
        if (self.algorithm in ['momentum', 'nag']):
            self.m_t = [[np.zeros(l.W.shape), np.zeros(l.b.shape)]  # weights, biases
                    for l in self.model.layers]

        if (self.algorithm in ['adam', 'nadam' , 'rmsprop']):
            self.m_t = [[np.zeros(l.W.shape), np.zeros(l.b.shape)]  # weights, biases
                        for l in self.model.layers]
            self.v_t = [[np.zeros(l.W.shape), np.zeros(l.b.shape)]  # weights, biases
                        for l in self.model.layers]
            self.iter=1

        self.__dict__.update(kwargs)

    
    def backpropagate(self, y_true, y_pred, loss):
        '''Calculates and returns gradients for each layer.'''
        gradients, batch_size = [], y_pred.shape[0]
        _, loss_grad = loss

        layers = self.model.layers.copy()
        
        if self.algorithm in ['nag']:  # lookahead
            for l in range(len(layers)):
                layers[l].W -= self.momentum * self.m_t[l][WEIGHT]
                layers[l].b -= self.momentum * self.m_t[l][BIAS]
        

        for l, layer in enumerate(reversed(layers)):
            
            if l == 0:  # output layer
                d_L = loss_grad(y_true, y_pred)
                d_A = np.multiply(d_L, layer.gradient())
                d_W = np.matmul(layer.input.T, d_A) / batch_size +  (self.weight_decay * layer.W) 
                d_b = np.mean(d_A, axis=0) if layer.use_bias else np.zeros(layer.b.shape)
            else:  # input and hidden layers
                d_A = np.multiply(np.matmul(d_A, next_layer.W.T), layer.gradient())
                d_W = np.matmul(layer.input.T, d_A) / batch_size +  (self.weight_decay * layer.W) 
                d_b = np.mean(d_A, axis=0) if layer.use_bias else np.zeros(layer.b.shape)

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
                    self.m_t[l][param] = ((self.momentum * self.m_t[l][param]) + (learning_rate * grad[param]))
                self.model.layers[l].W -= self.m_t[l][WEIGHT]
                self.model.layers[l].b -= self.m_t[l][BIAS]
                
        def nag(gradients, learning_rate):
            for l, grad in enumerate(gradients):
                for param in [WEIGHT, BIAS]:
                    self.m_t[l][param] = (self.momentum * self.m_t[l][param]) + (learning_rate * grad[param])
                self.model.layers[l].W -= self.m_t[l][WEIGHT]
                self.model.layers[l].b -= self.m_t[l][BIAS]

        def rmsprop(gradients, learning_rate):
            for l, grad in enumerate(gradients):
                for param in [WEIGHT, BIAS]:
                    self.v_t[l][param] = (self.beta * self.v_t[l][param]) + (1 - self.beta)*(np.square(grad[param]))
                self.model.layers[l].W -= learning_rate * (grad[WEIGHT]/np.sqrt(self.v_t[l][WEIGHT]+self.epsilon))
                self.model.layers[l].b -= learning_rate * (grad[BIAS] /np.sqrt(self.v_t[l][BIAS] + self.epsilon))

        def adam(gradients, learning_rate):

            for l, grad in enumerate(gradients):
                for param in [WEIGHT, BIAS]:
                    self.m_t[l][param] = (self.beta1 * self.m_t[l][param]) + (1 - self.beta1) * (grad[param])
                    self.v_t[l][param] = (self.beta2 * self.v_t[l][param]) + (1 - self.beta2)*(np.square(grad[param]))

                m_hat_w = (self.m_t[l][WEIGHT]) / (1 - (self.beta1) ** self.iter)
                v_hat_w = (self.v_t[l][WEIGHT]) / (1 - (self.beta2) ** self.iter)
                m_hat_b =  (self.m_t[l][BIAS]) / (1 - (self.beta1) ** self.iter)
                v_hat_b =  (self.v_t[l][BIAS]) / (1 - (self.beta2) ** self.iter)

                self.model.layers[l].W -= learning_rate * (m_hat_w / np.sqrt(v_hat_w + self.epsilon))
                self.model.layers[l].b -= learning_rate * (m_hat_b / np.sqrt(v_hat_b + self.epsilon))
                self.iter += 1

        def nadam(gradients, learning_rate):

            for l, grad in enumerate(gradients):
                for param in [WEIGHT, BIAS]:
                    self.m_t[l][param] = (self.beta1 * self.m_t[l][param]) + (1 - self.beta1) * (grad[param])
                    self.v_t[l][param] = (self.beta2 * self.v_t[l][param]) + (1 - self.beta2) * (np.square(grad[param]))

                m_hat_w = (self.m_t[l][WEIGHT]) / (1 - (self.beta1) ** self.iter)
                v_hat_w = (self.v_t[l][WEIGHT]) / (1 - (self.beta2) ** self.iter)
                m_hat_b = (self.m_t[l][BIAS]) / (1 - (self.beta1) ** self.iter)
                v_hat_b = (self.v_t[l][BIAS]) / (1 - (self.beta2) ** self.iter)

                m_dash_w = (self.beta1 * m_hat_w) + (1 - self.beta1) * grad[WEIGHT]
                m_dash_b = (self.beta1 * m_hat_b) + (1 - self.beta1) * grad[BIAS]
                self.model.layers[l].W -= learning_rate * (m_dash_w / np.sqrt(v_hat_w + self.epsilon))
                self.model.layers[l].b -= learning_rate * (m_dash_b / np.sqrt(v_hat_b + self.epsilon))
                self.iter += 1
                
        self.__dict__.update(kwargs)

        optimizer = {
            'sgd': sgd,
            'momentum': momentum,
            'nag': nag,
            'rmsprop': rmsprop,
            'adam': adam,
            'nadam': nadam
        }
        optimizer[self.algorithm](gradients, learning_rate, **kwargs)        
