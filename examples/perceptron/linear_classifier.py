from pyAI.models.perceptron import Perceptron
from pyAI.utils.utils import logo
import numpy as np 
import matplotlib.pyplot as plt
import argparse
import data

#================================
# Preprocessing
#================================

# Display logo
logo()

#================================
# Generate Training Set
#================================
training_set = data.generate_training_data(1000)

#================================
# Build SLP
#================================

# Instantiate perceptron object
slp = Perceptron(num_inputs=3)

# Train
lines = []
slp.training(training_set, lines)

# View training history
slp.plotTrainingHistory(training_set, lines)

#===================================
# Display training results
#===================================
slope = -slp.weights[0]/slp.weights[1]
intercept = -slp.weights[2]/slp.weights[1]

print("=============================================================")
print(f"Correct decision boundary: y = -1.00x + 45.00")
print("=============================================================")
print(f"Trained perceptron decision boundary: y = {slope:.2f}x + {intercept:.2f}")
print("=============================================================")