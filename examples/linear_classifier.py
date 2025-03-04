from pyAI.models.perceptron import Perceptron
import numpy as np 
import matplotlib.pyplot as plt
import training

#================================
# Generate Training Set
#================================
training_set = training.generate_training_set(30)

#================================
# Build SLP
#================================

# Instantiate
slp = Perceptron(num_inputs=3)

# Train
lines = []
slp.training(training_set, lines)

# View training history
slp.plotTrainingHistory(training_set, lines)

"""
#================================
# Plot Training Set 
#================================
x_plus = []
y_plus = []
x_minus = []
y_minus = []

for data in training_set:
	if training_set[data] == 1:
		x_plus.append(data[0])
		y_plus.append(data[1])
	elif training_set[data] == -1:
		x_minus.append(data[0])
		y_minus.append(data[1])
    
fig = plt.figure()
ax = plt.axes(xlim=(-25, 75), ylim=(-25, 75))

plt.scatter(x_plus, y_plus, marker = '+', c = 'green', s = 128, linewidth = 2)
plt.scatter(x_minus, y_minus, marker = '_', c = 'red', s = 128, linewidth = 2)

plt.title("Training Set")

plt.show()

print ('Weights: ', slp.weights)
print ('Number of Inputs: ', slp.num_inputs)
"""