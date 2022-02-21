import wandb
import argparse
import numpy as np
from tqdm import tqdm
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split 

from neural_networks import NeuralNetwork, Optimizer, clip_gradients, losses
from utils import preprocess_data, make_batches, accuracy, gradient_sum


def train_and_evaluate(args):

    # Setup wandb.ai
    run_name = '_'.join([f'e={args.epochs}', f'b={args.batch_size}', f'o={args.optimizer}',
        f'lr={args.learning_rate}', f'hl={args.num_layers}', f'sz={args.hidden_size}', f'a={args.activation}',
        f'w_i={args.weight_init}', f'w_d={args.weight_decay}',
        f'l={"".join([w[0] for w in args.loss.split("_")])}'  # extract loss short form
    ])
    print("Running: ", run_name)
    wandb.init(project=args.project_name, name=run_name)

    # Load Data
    (X_train, y_train), (X_test, y_test) = mnist.load_data() if args.dataset == 'mnist' else fashion_mnist.load_data()
    X_train,X_val,y_train,y_val = train_test_split(X_train, y_train, test_size=0.1,
                                                   random_state=1, stratify=y_train)
    (X_train, y_train) = preprocess_data(X_train, y_train)
    (X_test, y_test) = preprocess_data(X_test, y_test)
    (X_val,y_val) = preprocess_data(X_val,y_val)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape,X_val.shape,y_val.shape)

    # Instantiate model
    architecture = [{'num_neurons': args.hidden_size, 'activation': args.activation, 'init_method': args.weight_init}
                    for _ in range(args.num_layers)]
    architecture.append({'num_neurons': 10, 'activation': 'softmax'})  # add output layer

    model = NeuralNetwork(input_dim=784, architecture=architecture)
    optimizer = Optimizer(model, algorithm=args.optimizer,
                          momentum=args.momentum, beta=args.beta,
                          beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon,
                          weight_decay=args.weight_decay)
    lr = args.learning_rate
    loss_fn, _ = losses[args.loss]

    # Train

    for epoch in range(args.epochs):
        running_loss, running_grad = .0, .0
        num_steps = (X_train.shape[0] // args.batch_size)
        progress_bar = tqdm(make_batches(X_train, y_train, args.batch_size),
                            total=num_steps)
        for (X_batch, y_batch) in progress_bar:
            # Forward
            y_pred = model.forward(X_batch)

            # Optimize
            gradients = optimizer.backpropagate(y_batch, y_pred, losses[args.loss])
            gradients = clip_gradients(gradients, clip_ratio=5.0, norm_type=2)
            optimizer.optimize(gradients, learning_rate=lr)

            # Track acc, loss and gradients
            loss = loss_fn(y_batch, y_pred)
            grad_sum = gradient_sum(gradients)
            acc = accuracy(y_batch, y_pred)
            progress_bar.set_description(
                f"epoch: {epoch}, lr: {lr:.5f} | loss: {loss:.4f}, acc(batch): {acc:.4f}, grad:{grad_sum:.4f}"
            )
            
            running_loss += loss
            running_grad += grad_sum
        
        # Evaluate train and test splits
        train_acc = accuracy(model.forward(X_train), y_train)
        y_val_pred = model.forward(X_val)
        val_acc = accuracy(y_val_pred,y_val)
        val_loss = loss_fn(y_val,y_val_pred)
        test_acc = accuracy(model.forward(X_test), y_test)
        print(f"acc(train): {train_acc:.4f}, acc(val): {val_acc:.4f}, acc(test): {test_acc:.4f}")
        print('_' * 99)
        
        # Log metrics to wandb.ai
        wandb.log({
            'train_acc': train_acc, 
            'val_acc': val_acc,
            'train_loss': running_loss/num_steps,
            'val_loss' : val_loss,
            'test_acc': test_acc,
            'epoch':epoch            
        })
    
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.forward(X_test), axis=1)
    class_names = [str(i) for i in range(10)]
    wandb.log({
      "confusion_matrix" : wandb.plot.confusion_matrix(probs=None,
                              y_true=y_true, preds=y_pred,
                              class_names=class_names)
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project_name', type=str,
                        default='neural-networks-fashion-mnist',
                        help='Project name used to setup wandb.ai dashboard.')
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',
                        choices=['mnist', 'fashion_mnist'],
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
    parser.add_argument('-w_d', '--weight_decay', type=float, default=.0,
                        help='Weight decay used by optimizers'),
    parser.add_argument('-w_i', '--weight_init', type=str, default='Xavier_normal',
                        choices=['Xavier_normal', 'Xavier_uniform', 'He_normal', 'He_uniform'],
                        help='Weight initialization method used by optimizers')
    parser.add_argument('-nhl', '--num_layers', type=int, default=1,
                        help='Number of feedforward layers')
    parser.add_argument('-sz', '--hidden_size', type=int, default=32,
                        help='Number of hidden neurons in a feedfoward layer')
    parser.add_argument('-a', '--activation', type=str, default='ReLU',
                        choices=['identity', 'sigmoid', 'tanh', 'ReLU'],
                        help='Activation function of feedforward layer')
    args = parser.parse_args()
    
    train_and_evaluate(args)