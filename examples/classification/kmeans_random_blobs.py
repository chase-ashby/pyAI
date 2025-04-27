import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyAI.models.kmeans import KMeans
from pyAI.utils.utils import logo

# Display logo
logo()

# =======================================
# Generate random data
# =======================================
X, y = make_blobs(n_samples=1000, centers=4, cluster_std=0.80, random_state=0)

# =======================================
# Instantiate K means clustering object
# =======================================

model = KMeans(n_clusters=4, max_iter=1000, tol=0.0000001, random_state=0)

# =======================================
# Classify data into clusters
# =======================================
model.fit(X)

# =======================================
# Plot results
# =======================================
plt.scatter(X[:, 0], X[:, 1], c=model.labels, cmap='viridis', edgecolor='k')
plt.scatter(model.centroids[:, 0], model.centroids[:, 1],
            c='cyan', marker='*', s=200, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
