from . import Layer

def infer_layer_dimensions(input_dim, architecture):
    architecture_clean, prev_layer_dim = [], 0
    for l, layer in enumerate(architecture):
        layer['input_dim'] = prev_layer_dim if l != 0 else input_dim
        prev_layer_dim = layer['num_neurons']
        architecture_clean.append(layer)
    return architecture_clean


class NeuralNetwork():
    '''
    Implements a layer for a feed-forward neural network.
    To account for bias, augment input data with 1's.

    Input: (batch_size, dimension)
    Output: (batch_size, num_output_neurons)
    '''

    def __init__(self, input_dim, architecture):
        self.layers = []
        for layer_description in infer_layer_dimensions(input_dim, architecture):
            self.layers.append(Layer(**layer_description))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
