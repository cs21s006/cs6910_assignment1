import argparse
import numpy as np
from tqdm import tqdm
from keras.datasets import fashion_mnist

from neural_networks import NeuralNetwork, Optimizer, clip_gradients, losses
from utils import preprocess_data, make_batches, accuracy, gradient_sum


def train_and_evaluate(args):
    # Load Data
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    (X_train, y_train) = preprocess_data(X_train, y_train)
    (X_test, y_test) = preprocess_data(X_test, y_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


    # Instantiate model
    architecture = [{'num_neurons': args.hidden_size, 'activation': args.activation}
                    for _ in range(args.num_layers)]
    architecture.append({'num_neurons': 10, 'activation': 'softmax'})  # add output layer

    model = NeuralNetwork(input_dim=784, architecture=architecture)
    optimizer = Optimizer(model, algorithm=args.optimizer,
                          momentum=args.momentum, beta=args.beta,
                          beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)
    lr = args.learning_rate
    loss_fn, _ = losses[args.loss]

    # Train
    history = {'loss': [], 'gradient_sum': []}

    for epoch in range(args.epochs):

        progress_bar = tqdm(make_batches(X_train, y_train, args.batch_size),
                            total=(X_train.shape[0] // args.batch_size))
        for (X_batch, y_batch) in progress_bar:
            # Forward
            y_pred = model.forward(X_batch)

            # Optimize
            gradients = optimizer.backpropagate(y_batch, y_pred, losses[args.loss])
            gradients = clip_gradients(gradients, clip_ratio=5.0, norm_type=2)
            optimizer.optimize(gradients, learning_rate=lr)

            # Track acc, loss and gradients
            loss = loss_fn(y_batch, y_pred)
            history['loss'].append(loss)
            history['gradient_sum'].append(gradient_sum(gradients))
            acc = accuracy(y_batch, y_pred)
            progress_bar.set_description(
                f"epoch: {epoch}, lr: {lr:.5f} | loss: {loss:.4f}, acc(batch): {acc:.4f}, grad:{history['gradient_sum'][-1]:.4f}"
            )


        # Evaluate train and test splits
        train_acc = accuracy(model.forward(X_train), y_train)
        test_acc = accuracy(model.forward(X_test), y_test)
        print(f"acc(train): {train_acc:.4f}, acc(test): {test_acc:.4f}")
        print('_' * 99)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project_name', type=str,
                        default='neural-networks-fashion-mnist',
                        help='Project name used to setup wandb.ai dashboard.')
    parser.add_argument('-e', '--epochs', type=int, default=30,
                        help='Number of epochs to train model.')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='Batch size to be used to train and evaluate model')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'mean_squared_error', 'mean_absolute_error'],
                        help='Loss function used to optimize model parameters.')
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4,
                        help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='Momentum used by momentum and nag optimizers')
    parser.add_argument('-beta', '--beta', type=float, default=0.9,
                        help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9,
                        help='Beta1 used by adam and nadam optimizers')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.99,
                        help='Beta2 used by adam and nadam optimizers')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-6,
                        help='Epsilon used by optimizers')                                                
    parser.add_argument('-hl', '--num_layers', type=int, default=1,
                        help='Number of feedforward layers')
    parser.add_argument('-sz', '--hidden_size', type=int, default=32,
                        help='Number of hidden neurons in a feedfoward layer')
    parser.add_argument('-a', '--activation', type=str, default='ReLU',
                        choices=['identity', 'sigmoid', 'tanh', 'ReLU'],
                        help='Activation function of feedforward layer')
    args = parser.parse_args()

    run_name = '_'.join([
        f'e={args.epochs}',
        f'b={args.batch_size}',
        f'o={args.optimizer}',
        f'lr={args.learning_rate}',
        f'hl={args.num_layers}',
        f'sz={args.hidden_size}',
        f'a={args.activation}',
        f'l={"".join([w[0] for w in args.loss.split("_")])}'  # extract loss short form
    ])
    print(run_name)
    
    train_and_evaluate(args)