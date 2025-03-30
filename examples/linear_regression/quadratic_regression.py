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
m_examples = 10000
n_feautres = 2

# Random first column
column_1 = np.random.rand(m_examples, 1)

# Square first feature to generate second column
column_2 = column_1**2
column_2.reshape(m_examples, 1)

# Stack columns to create data matrix
X = np.hstack((column_1, column_2))

y = 0.1 * X[:, 0] + 100 * X[:, 1] + 1 + 0.5 * np.random.randn(m_examples)
y = y.reshape(m_examples, 1)

print('X shape:', X.shape)
print('y shape: ', y.shape)

# ================================
# Train Model
# ================================

# Instantiate model
model = LinearRegression(method='ne')

# Train model
model.fit(X, y)

# ================================
# Visualize Model
# ================================
model.plot_model(X, y, fit_type='poly')
