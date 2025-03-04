import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Perceptron:

  def __init__(self, num_inputs=3, weights=None):
    ''' 
    Initialize the Perceptron with a given number of inputs and optional weights.
    If weights are not provided, they default to ones.
    
    Parameters:
    - num_inputs (int): Number of input features.
    - weights (list or numpy array, optional): Initial weights for the perceptron.
    '''
    
    # Initialize attributes
    self.num_inputs = num_inputs

    if (weights is None):
        self.weights = np.ones(self.num_inputs)
    else:
        self.weights = np.array(weights)

    # Check dimension
    if (self.weights.shape[0] != self.num_inputs):
        raise ValueError(f"Number of inputs {self.num_inputs} does not match weights size {self.weights.shape}")
    
  def weighted_sum(self, inputs):
    '''
    Compute the weighted sum of input features.
    
    Parameters:
    - inputs (numpy array): Input feature vector.
    
    Returns:
    - float: Weighted sum of inputs.
    '''
    return np.dot(inputs, self.weights)

  def signed_activation(self, weighted_sum):
    ''' 
    Apply the signed activation function.
    
    Parameters:
    - weighted_sum (float): Weighted sum from the perceptron.
    
    Returns:
    - int: Activation result (1 if non-negative, -1 otherwise).
    '''
    return 1 if weighted_sum >= 0 else -1

  def training(self, training_set, lines=None, tol=1e-6):
    '''
    Train the perceptron using a simple weight update rule based on classification errors.
    
    Parameters:
    - training_set (dict): Mapping of input feature tuples to their respective class labels (1 or -1).
    - lines (list, optional): Stores decision boundary updates for visualization.
    - tol (float, optional): Tolerance for stopping criteria (not currently used).
    '''
    # Set maximum and minimum data range
    if (lines != None):
      xmin = min(data[0] for data in training_set)
      xmax = max(data[0] for data in training_set)

    foundLine = False
    while not foundLine:
      total_error = 0
      for inputs in training_set:
        input_data = np.array(inputs)
        prediction = self.signed_activation(self.weighted_sum(input_data))
        actual = training_set[inputs]
        error = actual - prediction
        total_error += abs(error)
        for i in range(self.num_inputs):
             self.weights[i] += 0.1 * (error * inputs[i])

        if (lines != None):
          # Generate line information

          # Slope and intercept
          w1 = self.weights[1]
          if (abs(self.weights[1]) < 1e-16): w1 = 1e-16
 
          slope = -self.weights[0]/w1
          intercept = -self.weights[2]/w1

          # Starting and endpoints of the line
          y1 = (slope * xmin) + intercept
          y2 = (slope * xmax) + intercept

          # Add to list for tracking training history
          lines.append([[xmin,xmax], [y1, y2]])

      if(total_error == 0): foundLine = True

  def plotTrainingHistory(self, training_set, lines):
    """
    Visualize the perceptron’s training process using an animated plot.
    
    Parameters:
    - training_set (dict): Mapping of input feature tuples to their respective class labels (1 or -1).
    - lines (list): List of decision boundary updates from training.
    """

    #===========================
    # Generate initial figure
    #===========================

    # Extract training data coordinates
    x_minus = []; x_plus=[]
    y_minus = []; y_plus=[]

    for data in training_set:
        if training_set[data] == 1:
            x_plus.append(data[0])
            y_plus.append(data[1])
        elif training_set[data] == -1:
            x_minus.append(data[0])
            y_minus.append(data[1])

    # Create figure
    fig = plt.figure()
    ax = plt.axes(xlim=(-25, 75), ylim=(-25, 75))
    line, = ax.plot([], [], lw=2)

    # Scatter plot for training data
    ax.scatter(x_plus, y_plus, marker = '+', c = 'green', s = 128, linewidth = 1)
    ax.scatter(x_minus, y_minus, marker = '_', c = 'red', s = 128, linewidth = 1)

    # Total number of iterations
    nIters = len(lines)-1

    # Create animation
    def animate(i):
        
        # Update the x and y data
        line.set_xdata(lines[i][0]) 
        line.set_ydata(lines[i][1])

        # Update the title to track iterations
        ax.set_title(f'Iteration: {i} of {nIters}')

        return line,

    def init():
        line.set_data([], [])
        return line,

    # Animation function
    a = animation.FuncAnimation(fig, animate, frames=len(lines), init_func=init, interval=0.001, blit=False, repeat=False)
    plt.show()