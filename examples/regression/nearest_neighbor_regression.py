from pyAI.models.nn_regression import NNRegression
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


def generate_circular_clusters(n_samples_per_cluster=1000, cluster_separation=3, noise_std=0.3):
    """
    Generates data with two circular clusters for regression testing.

    Args:
        n_samples_per_cluster: Number of samples in each cluster.
        cluster_separation: Distance between the cluster centers.
        noise_std: Standard deviation of the noise added to the outputs.

    Returns:
        X: Data matrix (n_samples, 2).
        y: Output vector (n_samples,).
    """

    # Cluster centers
    center1 = np.array([0, 0])
    center2 = np.array([cluster_separation, 0])

    # Generate data for cluster 1
    angles1 = np.random.rand(n_samples_per_cluster) * 2 * np.pi
    radii1 = np.random.rand(n_samples_per_cluster) + 1

    x1 = radii1 * np.cos(angles1) + center1[0]
    y1 = radii1 * np.sin(angles1) + center1[1]

    # Generate data for cluster 2
    angles2 = np.random.rand(n_samples_per_cluster) * 2 * np.pi
    radii2 = np.random.rand(n_samples_per_cluster) + 1
    x2 = radii2 * np.cos(angles2) + center2[0]
    y2 = radii2 * np.sin(angles2) + center2[1]

    # Combine data from both clusters
    X = np.concatenate([np.stack([x1, y1], axis=1),
                       np.stack([x2, y2], axis=1)])
    y = np.concatenate([radii1, radii2]) + \
        np.random.randn(2 * n_samples_per_cluster) * noise_std

    n = y.shape[0]
    y = y.reshape(n, 1)

    return X, y


X, y = generate_circular_clusters(
    n_samples_per_cluster=100, cluster_separation=10)

# ================================
# Setup Model
# ================================

# Setup NN model
model = NNRegression(X, y)

# ================================
# Generate test data
# ================================

X_test, y_test = generate_circular_clusters(
    n_samples_per_cluster=10, cluster_separation=10)

# Test against training matrix
predictions = model.predict(X_test)

plt.title('Nearest Neighbor Regression')
plt.xlabel('Feature')
plt.ylabel('Output')
plt.scatter(X[:, 0], model.targets, color='b', label='training')
plt.scatter(X_test[:, 0], y_test, color='k', alpha=0.3,
            marker='v', s=100, label='test')
plt.scatter(X_test[:, 0], predictions, color='r', alpha=0.3,
            marker='s', s=100, label='predictions')
plt.legend()
plt.show()
