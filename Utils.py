"""
Module containing all the Utility functions that is needed for the neural network.
"""

from enum import Enum
import numpy as np


class ActivationFunction(Enum):
  SIGMOID = "sigmoid"
  RELU = "ReLU"
  TAN_H = "tanh"
  IDENTITY = "identity"

class InitializationMethod(Enum):
  UNIFORM_RANDOM = "random"
  UNIFORM_XAVIER = "Xavier"
  GAUSSIAN_XAVIER = "gaussian_xavier"

class OutputFunction(Enum):
  SOFTMAX = "softmax"

class OptimizationAlgorithm(Enum):
  GD = "gd"
  SGD = "sgd"
  MINI_BATCH = "mini_batch"
  MOMENTUM_GD = "momentum"
  NAG = "nag"
  RMS_PROP = "rmsprop"
  ADAM = "adam"
  NADAM = "nadam"

class ErrorCalculationMethod(Enum):
  CROSS_ENTROPY = "cross_entropy"
  MEAN_SQUARE_ERROR = "mean_squared_error"


def activation_function(a, func):
  """
  Calculates post activation values from pre-activation values, functions implemented:
  * Sigmoid
  * ReLU
  * Tan h

  Parameters:
  -----------
  a: ndarray, pre-activation values
  func: Enum describing the activation function type

  Retruns:
  -------
  Post activation values in ndarray of the same dimention
  """
  if(func == ActivationFunction.SIGMOID):  
    new = a.copy()
    new[a<0] = np.exp(a[a<0])/(1.0 + np.exp(a[a<0]))
    new[a>=0] = 1/(1+np.exp(-a[a>=0]))
    return new

  if(func == ActivationFunction.RELU):
    return np.maximum(0,a)

  if(func == ActivationFunction.TAN_H):
    return np.tanh(a)
  
  if(func == ActivationFunction.IDENTITY):
    return a


def df_activation_function(a, func):
  """
  Calculates the derivative of the activation function, functions implemented:
  * Sigmoid
  * ReLU
  * Tan h
  
  Parameters:
  -------
  a: ndarray, pre-activation values
  func: Enum describing the activation function type
  
  """
  if(func == ActivationFunction.SIGMOID): 
    return activation_function(a, func) * (1 - activation_function(a, func))

  if(func == ActivationFunction.RELU):
    result = a.copy()
    result[result>=0] = 1
    result[result<0] = 0
    return result
  
  if(func == ActivationFunction.TAN_H):
    return 1 - np.square(activation_function(a, func))
  
  if(func == ActivationFunction.IDENTITY):
    return np.ones(shape=a.shape)


def output_function(a, func):
  """
  Given the pre-activation values, returns post activation values of the output layer
  """
  if(func == OutputFunction.SOFTMAX): 
    ones_array = np.ones((a.shape[1],a.shape[0]))
    numerator = np.exp(a - (ones_array * a.max(axis=1)).T)
    denominator = 1/numerator.sum(axis = 1) * ones_array 
    return denominator.T * numerator
  

def calc_total_error(predicted_distribution, true_label, method, weight_decay_for_l2_reg, weights):
  """Calculates the total error based on the error calculation method
  
  Params:
  --------
  predicted_distribution:
    ndarray containing the predicted probability distribution for each input
  true label:
    ndarray containing the true label for each inputs
  method:
    Enum describing the type of error calculation method used
  weight_decay_for_l2_reg:
    weight decay used 
  weights: 
    ndarray containings weights at each layer 

  Returns:
  -----
  L2 normalized error of the specified category
  """
  rows = np.arange(true_label.shape[0]) #setting row number from 0 to length(true label)
  cols = true_label
  L = 0.0
  if(method == ErrorCalculationMethod.CROSS_ENTROPY):
    predicted_distribution = predicted_distribution[rows,cols]
    predicted_distribution[predicted_distribution == 0] = 1e-6  #setting 0 values to very small value, so we dont get inf 
    L = sum(-np.log(predicted_distribution)) 

  if(method == ErrorCalculationMethod.MEAN_SQUARE_ERROR):
    predicted_distribution[rows,cols] -= 1
    L = np.sum(predicted_distribution ** 2)
  
  #starting from index:1 because we are not using index:0
  val = 0
  for i in range(1, len(weights)):
    val += np.sum(weights[i] ** 2)

  return L + (weight_decay_for_l2_reg / 2)*val
