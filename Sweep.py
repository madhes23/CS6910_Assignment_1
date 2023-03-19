from Utils import *
from NN import Classification
import wandb
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split


#downloading and splitting the data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size=0.8)
labels = ["T-shirt/top","Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#pre-processing the data
no_of_pixels = x_train[0].size
x_train = x_train.reshape(-1, no_of_pixels)/255
x_test = x_test.reshape(-1, no_of_pixels)/255
x_validation = x_validation.reshape(-1, no_of_pixels)/255


wandb.login()
sweep_config = {
    "method": 'random',
    "metric": {
    'name': 'accuracy',
    'goal': 'maximize'
    },
    'parameters' :{
        "hidden_layers": {"values":[2,3,4,5]},
        "neurons_per_hidden_layer": {"values": [32,64,128]},
        "learning_rate": {"values":[1e-3,1e-4]},
        "max_epoch": {"values":[5,10,15]},
        "batch_size": {"values":[16,32,64]},
        "activation_function" : {"values" : ["identity","sigmoid", "ReLU", "tanh"]},
        "optimization_algorithm": {"values":["sgd","mini_batch", "momentum", "nag", "rmsprop", "adam", "nadam"]},
        "initialization_method" : {"values" : ["random", "Xavier"]},
        "weight_decay" : {"values":[0, 0.0005, 0.05, 0.5]}
    }
}

def tune_nn():
    """A utility function for performing the sweep"""
    wandb.init()
    name = "{}_{}_hl_{}_with_{}_neurons_lr_{}_batch_{}_init_{}_l2_{}".format(
        wandb.config.optimization_algorithm,
        wandb.config.activation_function,
        wandb.config.hidden_layers,
        wandb.config.neurons_per_hidden_layer,
        wandb.config.learning_rate,
        wandb.config.batch_size,
        wandb.config.initialization_method,
        wandb.config.weight_decay
    )
    wandb.run.name = name
    no_of_hidden = wandb.config.hidden_layers 
    layer_size = wandb.config.neurons_per_hidden_layer
    hidden_layer = [layer_size] * no_of_hidden

    model = Classification(_no_of_class=10,
                           _hidden_layer = hidden_layer,
                           _input_layer = 784,
                           _max_epoch = wandb.config.max_epoch,
                           _activation_fun = ActivationFunction(wandb.config.activation_function),
                           _initialization_method = InitializationMethod(wandb.config.initialization_method),
                           _learning_rate = wandb.config.learning_rate,
                           _batch_size = wandb.config.batch_size,
                           _optimization_algorithm = OptimizationAlgorithm(wandb.config.optimization_algorithm),
                           _weight_decay_l2_reg = wandb.config.weight_decay)

    model.init_weight_and_bias()
    tr_err, val_err, tr_acc, val_acc = model.fit(x_train, y_train, x_validation, y_validation)

    for i in range(len(tr_err)):
      wandb.log({"tr_err":tr_err[i],
                 "tr_acc" : tr_acc[i],
                 "val_err" : val_err[i],
                 "val_acc" : val_acc[i],
                 "epoch":(i+1)})
      
    wandb.log({"accuracy": val_acc[-1]})


sweep_id=wandb.sweep(sweep_config,project="CS6910_Assignment-1")
wandb.agent(sweep_id,function=tune_nn,count=1)
wandb.finish()