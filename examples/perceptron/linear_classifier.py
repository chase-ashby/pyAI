"""
MIT License

Copyright (c) 2025 Chase Ashby

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from pyAI.models.perceptron import Perceptron
from pyAI.utils.utils import logo
import numpy as np
import matplotlib.pyplot as plt
import argparse
import data

# ================================
# Preprocessing
# ================================

# Display logo
logo()

# ================================
# Generate Training Set
# ================================
training_set = data.generate_training_data(1000)

# ================================
# Build SLP
# ================================

# Instantiate perceptron object
slp = Perceptron(num_inputs=3)

# Train
lines = []
slp.training(training_set, lines)

# View training history
slp.plotTrainingHistory(training_set, lines)

# ===================================
# Display training results
# ===================================
slope = -slp.weights[0]/slp.weights[1]
intercept = -slp.weights[2]/slp.weights[1]

print("=============================================================")
print(f"Correct decision boundary: y = -1.00x + 45.00")
print("=============================================================")
print(
    f"Trained perceptron decision boundary: y = {slope:.2f}x + {intercept:.2f}")
print("=============================================================")
