from pyAI.models.linear_regression import LinearRegression
from pyAI.utils.utils import logo
import numpy as np
import matplotlib.pyplot as plt

# ================================
# Preprocessing
# ================================

# Display logo
logo()

# ================================
# Generate Training Data
# ================================
m_examples = 1000
n_features = 2
X = np.random.rand(m_examples, n_features)

if (n_features == 1):
    y = 2 * X + 1 + 0.1 * np.random.randn(m_examples, n_features)
elif (n_features == 2):
    y = 0.1 * X[:, 0] + 100 * X[:, 1] + 1 + 0.1 * np.random.randn(m_examples)
    y = y.reshape(m_examples, 1)

# ================================
# Train Model
# ================================

# Instantiate model
model = LinearRegression(
    iterations=1000, learning_rate=0.05, tolerance=1e-6, method='gd')

# Train model
model.fit(X, y)

# ================================
# Visualize Model
# ================================
model.plot_model(X, y)
model.plot_cost_history()
