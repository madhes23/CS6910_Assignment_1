import warnings
warnings.filterwarnings("ignore")

from NN import Classification
from Utils import *
import argparse
from sklearn.model_selection import train_test_split
import wandb
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Used for setting the model")
parser.add_argument('-wp', '--wandb_project', metavar="", required=False, type=str, default="CS6910_Assignment-1", help="Project name used to track experiments in Weights & Biases dashboard")
parser.add_argument('-we', '--wandb_entity', metavar="", required=False, type=str, default="madhes23", help="Wandb Entity used to track experiments in the Weights & Biases dashboard")
parser.add_argument('-d', '--dataset', metavar="", required=False, type=str, default="fashion_mnist",choices=["mnist", "fashion_mnist"], help="Dataset used for training the Neural Network")
parser.add_argument('-e', '--epochs', metavar="", required=False, type=int, default=15, help="Number of epochs to train neural network")
parser.add_argument('-b', '--batch_size', metavar="", required=False, type=int, default=16, help="Batch size used to train neural network")
parser.add_argument('-l', '--loss', metavar="", required=False, type=str, default="cross_entropy", choices= ["mean_squared_error", "cross_entropy"], help="Loss Function used in the neural network")
parser.add_argument('-o', '--optimizer', metavar="", required=False, type=str, default="adam", choices= ["sgd","mini_batch", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Gradient descent optimizer used in the neural network")
parser.add_argument('-lr', '--learning_rate', metavar="", required=False, type=float, default=0.0001, help="Learning rate used to optimize model parameters")
parser.add_argument('-m', '--momentum', metavar="", required=False, type=float, default=0.9, help="Momentum used by momentum and nag optimizers.")
parser.add_argument('-beta', '--beta', metavar="", required=False, type=float, default=0.99, help="Beta used by rmsprop optimizer. Named as 'decay_rate_for_update' in the NN code")
parser.add_argument('-beta1', '--beta1', metavar="", required=False, type=float, default=0.9, help="Beta1 used by adam and nadam optimizers. Named as 'momentum' in the NN code")
parser.add_argument('-beta2', '--beta2', metavar="", required=False, type=float, default=0.99, help="Beta2 used by adam and nadam optimizers. Named as 'decay_rate_for_update' in the NN code")
parser.add_argument('-eps', '--epsilon', metavar="", required=False, type=float, default=1e-8, help="Epsilon used by optimizers. Named as 'small_const' in the NN code")
parser.add_argument('-w_d', '--weight_decay', metavar="", required=False, type=float, default=0.0005, help="Weight decay used by optimizers.")
parser.add_argument('-w_i', '--weight_init', metavar="", required=False, type=str, default="Xavier", choices=["random", "Xavier"], help="Initialization method used for weights initialization") 
parser.add_argument('-nhl', '--num_layers', metavar="", required=False, type=int, default=2, help="Number of hidden layers used in feedforward neural network") 
parser.add_argument('-sz', '--hidden_size', metavar="", required=False, type=int, default=128, help="Number of hidden neurons in a feedforward layer")
parser.add_argument('-a', '--activation', metavar="", required=False, type=str, default="tanh", choices=["identity", "sigmoid", "tanh", "ReLU"], help="Activation function used in layers")
args = parser.parse_args()


if(__name__ == '__main__'):

    #downloading and splitting the data
    if(args.dataset == "fashion_mnist"):
        from keras.datasets import fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif (args.dataset == "mnist"):
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size=0.8)

    #pre-processing the data for Neural network
    no_of_pixels = x_train[0].size
    x_train = x_train.reshape(-1, no_of_pixels)/255
    x_test = x_test.reshape(-1, no_of_pixels)/255
    x_validation = x_validation.reshape(-1, no_of_pixels)/255



    decay_rate_for_update = 0
    if(args.optimizer == "adam" or args.optimizer == "nadam"):
        decay_rate_for_update = args.beta2
    else:
        decay_rate_for_update = args.beta

    hidden_layer= [args.hidden_size] * args.num_layers
    model = Classification(_hidden_layer = hidden_layer,
                           _max_epoch = args.epochs,
                           _batch_size = args.batch_size,
                           _error_calculation = ErrorCalculationMethod(args.loss),
                           _optimization_algorithm = OptimizationAlgorithm(args.optimizer),
                           _learning_rate = args.learning_rate,
                           _momentum = args.momentum,
                           _momentum_for_adam = args.beta1,
                           _decay_rate_for_update = decay_rate_for_update,
                           _small_const = args.epsilon,
                           _weight_decay_l2_reg = args.weight_decay,
                           _initialization_method = InitializationMethod(args.weight_init),
                           _activation_fun = ActivationFunction(args.activation))
    
    tr_err, val_err, tr_acc, val_acc = model.fit(x_train, y_train, x_validation, y_validation)
    print("Accuracy on test data: ", model.calc_accuracy(x_test, y_test))


    #Plotting the graph
    print("Plotting the graphs:")
    fig = model.plot_graphs(tr_err, val_err, tr_acc, val_acc)
    plt.show()


    #logging on wandb
    print("Syncing loss and accuracies to WandB: ")
    run_name = "command_line_{}_{}_{}L_with_{}_neur".format(args.optimizer, args.activation, args.num_layers, args.hidden_size)
    wandb.init(project = args.wandb_project, entity = args.wandb_entity, name = run_name)

    for i in range(len(tr_err)):
      wandb.log({"tr_err":tr_err[i],
                 "tr_acc" : tr_acc[i],
                 "val_err" : val_err[i],
                 "val_acc" : val_acc[i],
                 "epoch":(i+1)})
    
    wandb.finish()

