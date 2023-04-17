<a href="https://wandb.ai/madhes23/CS6910_Assignment-1/reports/Report-Assignment-1-CS6910---VmlldzozODI2Njkz" target="_blank">
  <img src="https://assets.website-files.com/5ac6b7f2924c656f2b13a88c/60de02f3843862bd5d482a51_weights-and-biases-logo-black.svg" alt="WandB logo" width="100">
  View report
</a>

# Exploring Deep learning
Learning the basic algorithms in feed forward neural network (while doing academic projects)

# Code structure
1. NN.py : contains the neural network class
2. Utils.py : contains all the Utility functions and Enums used in the NN.py
3. Sweep.py : code used for the Hyperparameter tuning using wandb sweeps
4. train.py : A utility file to configure the parameters of the model and run it

# How to initialize a model
1. Create a model in ```Models.ipynb``` like the following: 
    ```python
    model = Classification(_no_of_class = 10, 
                            _hidden_layer = [128, 128],
                            _input_layer = 784,
                            _max_epoch=15,
                            _optimization_algorithm=OptimizationAlgorithm.ADAM,
                            _activation_fun=ActivationFunction.TAN_H,
                            _initialization_method=InitializationMethod.UNIFORM_XAVIER,
                            _output_function = OutputFunction.SOFTMAX,
                            _error_calculation=ErrorCalculationMethod.CROSS_ENTROPY,
                            _learning_rate = 0.0001,
                            _batch_size = 16,
                            _momentum = 0.9,
                            _momentum_for_adam = 0.9,
                            _decay_rate_for_update = 0.99,
                            _weight_decay_l2_reg = 0.0005,
                            _small_const = 1e-8)
    ```
2. After initiating the parameters for the model, initiate Weights and Bias using
   ```python
   model.init_weight_and_bias()
   ```
3. Train the model, by passing the training and validation data (along with labels)
   ```python
    model.fit(x_train, y_train, x_validation, y_validation)
   ```
4. We can get the accuracy for the testing data using 
   ```python
   model.calc_accuracy(x_test, y_test)
   ```

# Expandability of the code
All the components have been included as modules. If you need to append an additional method to any of the following, we can add the new module to ```Utils.py```
* Activation algorithm
* Error Calculation method
* Initialization Method
* Output function
  
To add an extra optimization algorithm, there are two categories to consider. Some algorithms require lookahead, while others do not. Both types of algorithms have a separate place to be added to the code, which is properly commented for clarity.

When adding an algorithm that requires lookahead, please ensure that it is included in the relevant section of the code with the proper comments. Similarly, when adding an algorithm that does not require lookahead, make sure to place it in the correct section and comment appropriately.

By following these guidelines, new optimization algorithms can be easily incorporated into the existing code.
