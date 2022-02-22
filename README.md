# CS6910 Assignment 1

[Link to Weights & Biases Report](https://wandb.ai/cs21s006_cs21s043/cs6910_assignment1/reports/Assignment-1--VmlldzoxNTgzMjkz)

## Setup and Train on Fashion-MNIST

**Note:** It is recommended to create a new python virtual environment before installing dependencies.

```
pip install requirements.txt
python train.py
```

The number of hidden layers and the number of neurons in each hidden layer can be changed easily by passing command line arguments to the training script.

```
python train.py --num_layers 4 --hidden_size 256
```

### Arguments

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-p`, `--project_name` | neural-networks-fashion-mnist | Project name used to track experiments in Weights & Biases dashboard |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 30 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 16 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | adam | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.0003 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.9 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.9 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.9 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.99 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | Xavier_normal | choices:  ["Xavier_normal", "Xavier_uniform", "He_normal", "He_uniform"] | 
| `-nhl`, `--num_layers` | 1 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 32 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | ReLU | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |


## Examples, Usage and More

### Defining a Model

```python
from neural_networks import NeuralNetwork

architecture = [
    {'num_neurons': 784, 'activation': 'ReLU', 'init_method': 'Xavier_normal'},
    {'num_neurons': 256, 'activation': 'tanh', 'init_method': 'He_normal'},
    {'num_neurons': 10, 'activation': 'softmax'},
]
model = NeuralNetwork(input_dim=784, architecture=architecture)

# Forward pass
y_pred = model.forward(X)
```

### Defining Optimizer

```python
from neural_networks import Optimizer, losses

# Define an optimizer for a model
optimizer = Optimizer(model, algorithm='adam', 
                      beta1=0.9, beta2=0.99, epsilon=1e-8,
                      weight_decay=0.0005)

# Optimize model parameters
gradients = optimizer.backpropagate(y_true, y_pred, losses['cross_entropy'])
optimizer.optimize(gradients, learning_rate=3e-4)
```

### Defining a New Optimization Algorithm
To add a new optimisation algorithm, one can define a function for the new algorithm under the [`optimize` function](neural_networks/optimizer.py#L69) of `class Optimizer`. This function takes 2 parameters as input - 

```python
def new_optim_func(gradients: List[Tuple(np.ndarray, np.ndarray)], learning_rate: float):
    """
    Updates model weights and biases by accessing 
    self.model.layers[l].W and self.model.layers[l].b
    respectively.
    Input: 
       gradients - A list of tuples containing (d_W, d_b) 
                   corresponding to each layer.
       learning_rate - float
    Returns: None
    """
    pass
``` 
Other local optimizer variables required can be passed/ defined in the `__init__` method of `class Optimizer`.
Finally, the new optimisation algorithm can be registered by linking its name to the newly defined function in the [`optimizer` dictionary](https://github.com/cs21s006/cs6910_assignment1/blob/main/neural_networks/optimizer.py#L133). 


## Quick Links

* [Question 1](Question_1.ipynb)
* [Question 2, 3](neural_networks/) 
* [Question 4](Question_4.ipynb)
* [Question 5, 6, 7](https://wandb.ai/cs21s006_cs21s043/cs6910_assignment1/reports/Assignment-1--VmlldzoxNTgzMjkz)
* [Question 8](Question_8.ipynb)
* [Question 9](https://github.com/cs21s006/cs6910_assignment1)
* [Question 10](Question_10.ipynb)

## Team 
* [CS21S043](https://github.com/jainsaurabh426)
* [CS21S006](https://github.com/cs21s006)