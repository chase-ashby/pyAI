import numpy as np

class Perceptron:

  def __init__(self, num_inputs=2, weights=None):
    ''' 
    Constructor
    '''
    
    # Initialize attributes
    self.num_inputs = num_inputs

    if (weights is None):
        self.weights = np.ones(self.num_inputs)
    else:
        self.weights = np.array(weights)

    # Check dimension
    if (self.weights.shape[0] != num_inputs):
        raise ValueError(f"Number of inputs {self.num_inputs} does not match weights size {self.weights.shape}")
    
  def weighted_sum(self, inputs):
    '''
    Weighted summation of input data
    '''
    if (inputs.shape != self.weights.shape):
        raise ValueError(f"Input size {input_data.shape} does not match weights size {self.weights.shape}")

  
  def signed_activation(self, weighted_sum):
    ''' 
    Signed Activation Function
    '''

    sign = -1
    if (weighted_sum >= 0): sign = 1

    return sign