"""
A simple feed-forward Neural Network capable of classifying a given input (single dimentional) to a set of classes
"""

from Utils import *
import math
import numpy as np
import matplotlib.pyplot as plt


class Classification:
  """
  A simple feed-forward Neural Network capable of classifying a given input (single dimentional) to a set of classes
  """

  weight = []
  bias = []
  
  #Best Hyperparameters are initialized as default parameters
  def __init__(self, _no_of_class = 10, _hidden_layer = [128, 128], _input_layer = 784, _max_epoch=15, _optimization_algorithm=OptimizationAlgorithm.ADAM, _activation_fun=ActivationFunction.TAN_H, _initialization_method=InitializationMethod.UNIFORM_XAVIER, _output_function = OutputFunction.SOFTMAX, _error_calculation=ErrorCalculationMethod.CROSS_ENTROPY, _learning_rate = 0.0001, _batch_size = 16, _momentum = 0.9, _momentum_for_adam = 0.9, _decay_rate_for_update = 0.99, _weight_decay_l2_reg = 0.0005, _small_const = 1e-8) -> None:
    self.no_of_class = _no_of_class
    self.hidden_layer = _hidden_layer
    self.activation_func = _activation_fun
    self.max_epoch = _max_epoch
    self.input_layer = _input_layer
    self.output_func = _output_function
    self.learning_rate = _learning_rate
    self.initialization_method = _initialization_method
    self.error_calculation = _error_calculation
    self.optimization_algorithm = _optimization_algorithm
    self.batch_size = _batch_size
    self.momentum = _momentum
    self.decay_rate_for_update = _decay_rate_for_update
    self.weight_decay_l2_reg = _weight_decay_l2_reg
    self.small_const = _small_const
    self.momentum_for_adam = _momentum_for_adam

    #inferred values (for easier calculation)
    self.layer = [_input_layer]
    self.layer = self.layer + _hidden_layer
    self.layer.append(_no_of_class)    
    self.L = len(self.layer) - 1  #number of layers excluding the input layer (L value used in the lectures)


  def init_weight_and_bias(self):
    """
    Initiates the weights and bias matrices for the NN based on the initialization method specified.
    
    NOTE: A common convention is the 0-th index is some random value of shape (1,1), so we can follow the 
    1-based indexing taught in the class

    Example:
    ------
    1. Let the input layer = 784 neurons
    2. One hidden layer with 32 neurons
    3. Output layer has 10 neurons 

    Weight matrix shape is as follows:
    w[0] = (1,1) - some random number (refer NOTE)
    w[1] = (32, 784)
    w[2] = (10, 32)
    
    """
    #going to use 1-based indexing (as tought in the class)
    #so adding some random matrix in 0-th index
    w = [np.random.rand(1,1)]
    b = [np.random.rand(1,1)]
    if(self.initialization_method == InitializationMethod.UNIFORM_RANDOM): 
      low = -1
      high = 1
      i = 1
      while i < len(self.layer):
        w.append(np.random.uniform(low, high, size=(self.layer[i], self.layer[i-1])))
        b.append(np.zeros(self.layer[i]))
        i +=1
    
    if(self.initialization_method == InitializationMethod.UNIFORM_XAVIER): 
      for i in range(1, len(self.layer)):
        inputs = self.layer[i-1]
        outputs = self.layer[i]
        # x = math.sqrt(6/ inputs+outputs)
        x = math.sqrt(1/inputs)
        w.append(np.random.uniform(low=-x, high=x, size=(self.layer[i], self.layer[i-1])))
        b.append(np.zeros(self.layer[i]))

    if(self.initialization_method == InitializationMethod.GAUSSIAN_XAVIER): 
      mu = 0.0
      for i in range(1, len(self.layer)):
        inputs = self.layer[i-1]
        outputs = self.layer[i]
        # sigma = math.sqrt(6 / inputs+outputs)
        sigma = math.sqrt(1/ inputs)
        w.append(np.random.normal(mu, sigma, size=(self.layer[i], self.layer[i-1])))
        b.append(np.zeros(self.layer[i]))
    self.weight = np.array(w, dtype=object)
    self.bias = np.array(b, dtype=object)


  def init_prev_moments(self):
    """
    Returns the previous moments for both weight and bias (ie zeros) with same dimention as weights and biases

    Returns:
    ------
    prev_moment_w: ndarray, dtype=object
      All the values are initialized with zero, with same structure as weight matrix of the network
    prev_moment_b: ndarray, dtype=object
      All the values are initialized with zero, with same structure as bias matrix of the network
    """
    #going to use 1-based indexing (as tought in the class)
    #so adding some random matrix in 0-th index
    prev_moment_w = [np.random.rand(1,1)]
    prev_moment_b = [np.random.rand(1,1)]
    for i in range(1, len(self.layer)):
      prev_moment_w.append(np.zeros(shape=(self.layer[i], self.layer[i-1])))
      prev_moment_b.append(np.zeros(self.layer[i]))

    return np.array(prev_moment_w, dtype=object), np.array(prev_moment_b, dtype=object)

  def forward_propogation(self, input_images):
    """
    Performs feed forwarding (forward propogation) on the nerual network
    Preactivaion and post activation follows the method selected while initializing the class
    The final post activation (ie, post activation of output layer) follows Softmax

    Params:
    -----
    input_images : ndarray of shape (no_of_samples, pix), where pix is the number of pixels in the input image

    Returns:
    ------
    NOTE: A common convention is the 0-th index is some random value of shape (1,1), so we can follow the 
    1-based indexing taught in the class

    a : list of ndarrays contaning pre-activaton of each layers
    h : list of ndarrays contaning post-activaton of each layers
    """
    #for using 1 based indexing, adding some random matrix in 0-th index
    a = [np.random.rand(1,1)]
    h = [input_images]

    for i in range(1, self.L):
      a.append(self.bias[i] + np.dot(h[i-1], self.weight[i].T))
      h.append(activation_function(a[i], self.activation_func))

    a.append(self.bias[self.L] + np.dot(h[self.L-1], self.weight[self.L].T))
    h.append(output_function(a[-1], self.output_func))
    return a, h


  def backward_propogation(self, a, h, true_label, weight = None, use_custom_weight_matrix = False):
    """
    Params:
    ------
    a: list of ndarray
      pre activation values of the layers of the network (for the data points the in the batch)
    h: list of ndarray
      post activation values of the layers of the network (for the data points the in the batch)
    true_label: ndarray
      true label values of each of the data points in the batch
    weight: ndarray
      By default weight = weights populated in the network
      But we can pass a custom weight matrix to perfom back propogation

    Returns:
    -------
    NOTE: A common convention is the 0-th index is some random value of shape (1,1), so we can follow the 
    1-based indexing taught in the class

    del_w: ndarray of ndarrays contaning dereivating of w (same shape as w)
    del_b: ndarray of ndarrays contaning dereivating of b (same shape as b)
    """
    if(use_custom_weight_matrix == False):
      weight = self.weight
    
    del_a = [None] * (self.L+1)
    del_h = [None] * (self.L+1)
    del_w = [None] * (self.L+1)
    del_b = [None] * (self.L+1)
    
    #computing del_a_l
    del_a[-1] = h[-1].copy()
    row_ind = np.arange(true_label.shape[0]) #creating numbers 0 to batch size (for row indices)
    del_a[-1][row_ind,true_label] -= 1

    for k in range(self.L, 0, -1):
      #computing gradients w.r.t parameters
      del_w[k] = np.dot(del_a[k].T, h[k-1])
      del_b[k] = np.sum(del_a[k], axis=0)

      #computing gradients w.r.t layer below (post-activation)
      del_h[k-1] = np.dot(del_a[k],weight[k])

      #computing gradients w.r.t layer below (pre-activation)
      del_a[k-1] = del_h[k-1] * df_activation_function(a[k-1], self.activation_func)
    
    #setting the 0-th index to some random array of (1,1)
    #so that it won't cause dimention mismatch
    del_w[0] = np.random.rand(1,1)
    del_b[0] = np.random.rand(1,1)
    return np.array(del_w, dtype=object), np.array(del_b, dtype=object)


  def update_mini_batch(self, del_w, del_b):
    self.weight = self.weight - (self.learning_rate * del_w) - (self.learning_rate * self.weight_decay_l2_reg * self.weight)
    self.bias = self.bias - (self.learning_rate * del_b) - (self.learning_rate * self.weight_decay_l2_reg * self.bias)


  def update_momentum_gd(self, prev_moment_w, prev_moment_b, del_w, del_b):
    prev_moment_w = self.momentum * prev_moment_w + (1 - self.momentum) * del_w
    prev_moment_b = self.momentum * prev_moment_b + (1 - self.momentum) * del_b

    self.weight = self.weight - (self.learning_rate * prev_moment_w) - (self.learning_rate * self.weight_decay_l2_reg * self.weight)
    self.bias = self.bias - (self.learning_rate * prev_moment_b) - (self.learning_rate * self.weight_decay_l2_reg * self.bias)
    
    return prev_moment_w, prev_moment_b


  def update_nag(self, prev_moment_w, prev_moment_b, del_w_lookahead, del_b):
    prev_moment_w = self.momentum * prev_moment_w + (1 - self.momentum) * del_w_lookahead
    prev_moment_b = self.momentum * prev_moment_b + (1 - self.momentum) * del_b

    self.weight = self.weight - (self.learning_rate * prev_moment_w) - (self.learning_rate * self.weight_decay_l2_reg * self.weight)
    self.bias = self.bias - (self.learning_rate * prev_moment_b) - (self.learning_rate * self.weight_decay_l2_reg * self.bias)
    return prev_moment_w, prev_moment_b
  

  def update_rms_prop(self, prev_update_w, prev_update_b, del_w, del_b):
    prev_update_w = self.decay_rate_for_update * prev_update_w + (1 - self.decay_rate_for_update) * np.square(del_w)
    prev_update_b = self.decay_rate_for_update * prev_update_b + (1 - self.decay_rate_for_update) * np.square(del_b)

    new_lr_w = self.learning_rate / (prev_update_w**0.5 + self.small_const)
    new_lr_b = self.learning_rate / (prev_update_b**0.5 + self.small_const)

    self.weight = self.weight - (new_lr_w * del_w) - (self.learning_rate * self.weight_decay_l2_reg * self.weight)
    self.bias = self.bias - (new_lr_b * del_b) - (self.learning_rate * self.weight_decay_l2_reg * self.bias)
    return prev_update_w, prev_update_b


  def update_adam(self, x_train, y_train, prev_moment_w, prev_moment_b, prev_update_w, prev_update_b, del_w, del_b):
    prev_moment_w = self.momentum * prev_moment_w + (1 - self.momentum) * del_w
    prev_w_hat = prev_moment_w / (1 - self.momentum)
    prev_moment_b = self.momentum * prev_moment_b + (1 - self.momentum) * del_b
    prev_b_hat = prev_moment_b / (1 - self.momentum)

    prev_update_w = self.momentum * prev_update_w + (1 - self.momentum) * np.square(del_w)
    prev_update_m_hat = prev_update_w / (1 - self.momentum)
    prev_update_b = self.momentum * prev_update_b + (1 - self.momentum) * np.square(del_b)
    prev_update_b_hat = prev_update_b / (1 - self.momentum)

    new_lr_w = self.learning_rate / (prev_update_m_hat**0.5 + self.small_const)
    new_lr_b = self.learning_rate / (prev_update_b_hat**0.5 + self.small_const)

    self.weight = self.weight - (new_lr_w * prev_w_hat) - (self.learning_rate * self.weight_decay_l2_reg * self.weight)
    self.bias = self.bias - (new_lr_b * prev_b_hat) - (self.learning_rate * self.weight_decay_l2_reg * self.bias)

    return prev_moment_w, prev_moment_b, prev_update_w, prev_update_b


  def update_nadam(self, x_train, y_train, prev_moment_w, prev_moment_b, prev_update_w, prev_update_b, del_w, del_b):
    prev_moment_w = self.momentum * prev_moment_w + (1 - self.momentum) * del_w
    prev_w_hat = prev_moment_w / (1 - self.momentum)
    prev_moment_b = self.momentum * prev_moment_b + (1 - self.momentum) * del_b
    prev_b_hat = prev_moment_b / (1 - self.momentum)

    prev_update_w = self.decay_rate_for_update * prev_update_w + (1 - self.decay_rate_for_update) * np.square(del_w)
    prev_update_m_hat = prev_update_w / (1 - self.decay_rate_for_update)
    prev_update_b = self.decay_rate_for_update * prev_update_b + (1 - self.decay_rate_for_update) * np.square(del_b)
    prev_update_b_hat = prev_update_b / (1 - self.decay_rate_for_update)

    new_lr_w = self.learning_rate / (prev_update_m_hat**0.5 + self.small_const)
    new_lr_b = self.learning_rate / (prev_update_b_hat**0.5 + self.small_const)

    self.weight = self.weight - (new_lr_w * (self.momentum * prev_w_hat + (((1/self.momentum) * del_w) / (1-self.momentum)))) - (self.learning_rate * self.weight_decay_l2_reg * self.weight)
    self.bias = self.bias - (new_lr_b * (self.momentum * prev_b_hat + (((1/self.momentum) * del_b) / (1-self.momentum)))) - (self.learning_rate * self.weight_decay_l2_reg * self.bias)

    return prev_moment_w, prev_moment_b, prev_update_w, prev_update_b


  def fit(self, x_train, y_train, x_validation, y_validation): 
    """
    Trains and sets the weights and biases of the Neural Network

    Params:
    ------
    x_train: ndarray of shape (no_of_samples, pix), where pix is the number of pixels in the input image
    y_train: ndarray of shape (no_of_samples, ) represents the lables
    x_validation: ndarray of shape (no_of_validation_samples, pix), where pix is the number of pixels in the input image
    y_validation: ndarray of shape (no_of_validation_samples, ) represents the lables

    Retunrs:
    -----
    training_error_list: List contaning average errors (of training data) in each epochs
    validation_error_list: List contaning average errors (of validation data) in each epochs
    training_accuracy: List containing accuracy (of training data) in each epochs
    validation_accuracy: List containing accuracy (of validation data) in each epochs
    """
    self.data_size = x_train.shape[0]
    no_of_batches = self.data_size // self.batch_size
    self.init_weight_and_bias()
    prev_moment_w, prev_moment_b = self.init_prev_moments()
    prev_update_w, prev_update_b = prev_moment_w.copy(), prev_moment_b.copy()

    training_error_list = []
    validation_error_list = []
    training_accuracy = []
    validation_accuracy = []

    for i in range(self.max_epoch):
      err = 0
      for j in range(no_of_batches+1): 
        begin = j * self.batch_size
        end = begin + self.batch_size
        if(end > self.data_size):
          end = self.data_size

        a, h = self.forward_propogation(x_train[begin:end])

        #insert your algorithm which needs look ahead here:
        #----------------------------------------------
        if(self.optimization_algorithm == OptimizationAlgorithm.NAG):
          w_lookahead = self.weight - self.momentum * prev_moment_w
          del_w_lookahead, del_b = self.backward_propogation(a, h, y_train[begin:end],weight=w_lookahead, use_custom_weight_matrix=True)
          prev_moment_w, prev_moment_b = self.update_nag(prev_moment_w, prev_moment_b, del_w_lookahead, del_b)

        #inser your algorithm which doesn't need lookahead here:
        #---------------------------------------------------
        del_w, del_b = self.backward_propogation(a, h, y_train[begin:end])
        if(self.optimization_algorithm == OptimizationAlgorithm.MINI_BATCH):
          self.update_mini_batch(del_w, del_b)
        
        if(self.optimization_algorithm == OptimizationAlgorithm.SGD):
          self.update_mini_batch(del_w, del_b)
        
        if(self.optimization_algorithm == OptimizationAlgorithm.MOMENTUM_GD):
          prev_moment_w, prev_moment_b = self.update_momentum_gd(prev_moment_w, prev_moment_b, del_w, del_b)
        
        if(self.optimization_algorithm == OptimizationAlgorithm.NAG):
          prev_moment_w, prev_moment_b = self.update_nag(prev_moment_w, prev_moment_b, del_w_lookahead, del_b)
        
        if(self.optimization_algorithm == OptimizationAlgorithm.RMS_PROP):
          prev_update_w, prev_update_b = self.update_rms_prop(prev_update_w, prev_update_b, del_w, del_b)

        if(self.optimization_algorithm == OptimizationAlgorithm.ADAM):
          prev_moment_w, prev_moment_b, prev_update_w, prev_update_b = self.update_adam(x_train[begin:end], y_train[begin:end], prev_moment_w, prev_moment_b, prev_update_w, prev_update_b, del_w, del_b)

        if(self.optimization_algorithm == OptimizationAlgorithm.NADAM):
            prev_moment_w, prev_moment_b, prev_update_w, prev_update_b = self.update_nadam(x_train[begin:end], y_train[begin:end], prev_moment_w, prev_moment_b, prev_update_w, prev_update_b, del_w, del_b)

        err += calc_total_error(h[-1], y_train[begin:end], self.error_calculation, self.weight_decay_l2_reg, self.weight)
      err /= self.data_size
      training_error_list.append(err)
      _, h_val = self.forward_propogation(x_validation)
      validation_error_list.append(calc_total_error(h_val[-1], y_validation, self.error_calculation, self.weight_decay_l2_reg, self.weight)/ y_validation.size)
      acc = self.calc_accuracy(x_train, y_train)
      training_accuracy.append(acc)
      validation_accuracy.append(self.calc_accuracy(x_validation, y_validation))
      print("Completed epoch : {} \t Error: {} \t Accuracy: {}".format(i+1, err, acc))
    return training_error_list, validation_error_list, training_accuracy, validation_accuracy
     

  def calc_accuracy(self, x_test, y_test, return_predicted_distribution = False):
    """
    Calculates accuracy for x_test and y_test if return_predicted_distrubution = False
    if return_predicted_distrubution == True:
      Returns predicted distributions

    Params:
    ------
    x_test : ndarray of data 
    y_test : ndarray of labels
    return_predicted_distribution: bool (by default = False)

    """
    _ , h = self.forward_propogation(x_test)
    predicted_distribution = h[-1]
    if(return_predicted_distribution == True):
      return predicted_distribution
    return np.sum(np.argmax(predicted_distribution, axis=1) == y_test) / y_test.size
  
  def plot_graphs(self, training_errors, validation_errors, training_accuracy, validation_accuracy):
    """
    Plots a Error and Accuracy graphs for training and validation data over the epochs

    Params:
    -----
    training_errors: list containing error (ie loss values) of the training data over the epochs
    validation_errors: list containing error (ie loss values) of the validation data over the epochs
    training_accuracy: list containing accuracy of the training data over the epochs
    validation_accuracy: list containing accuracy of the validation data over the epochs

    Returns:
    -----
    fig: matplot figure object 
    """
    x = np.arange(len(training_errors))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x, training_errors, label = "Training Error")
    ax1.plot(x, validation_errors, label = "Validation Error")
    ax1.set_title("Errors")
    ax1.legend()
    ax2.plot(x, training_accuracy, label = "Training Accuracy")
    ax2.plot(x, validation_accuracy, label = "Validation Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    return fig