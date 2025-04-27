import numpy as np


class KMeans:
    """
    A class for performing K-Means clustering.
    """

    def __init__(self, n_clusters=8, max_iter=300, tol=0.0001, random_state=None):
        """
        Default constructor
        """

        if (n_clusters < 1):
            raise ValueError("Number of clusters must be >= 1")

        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def kmeans_pp_init(self, data):
        """
        kmeans++ initialization routine

        Args:
            data (numpy.ndarray): The data points, shape (n_samples, n_features).
        """

        # Set random seeding
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Number of samples and features
        n_samples = data.shape[0]
        n_features = data.shape[1]

        # Initialize centroids
        centroids = np.zeros((self.k, n_features))

        # ================================================
        # Choose 1st centroid uniformly at random from X
        # ================================================
        random_index = np.random.choice(n_samples)
        centroids[0] = data[random_index]
        print(centroids[0])
        # ================================================
        # Set remaining centroids
        # ================================================
        for iCentroid in range(1, self.k):

            # Compute minimum distance of each point to
            # the set of the set centroids
            distances = np.zeros(n_samples)
            for i in range(n_samples):
                min_dist = float('inf')
                for j in range(iCentroid):
                    dist = np.linalg.norm(data[i, :] - centroids[j, :])
                    min_dist = min(min_dist, dist)
                distances[j] = min_dist

            # Set probabilities for each point
            probabilities = distances**2 / np.sum(distances**2)

            # Randomly select a new centroid with probabilities
            # prescribed to ensure a centroid is selected that
            # is "far" away from the other centroids
            random_index = np.random.choice(n_samples, p=probabilities)
            centroids[iCentroid, :] = data[random_index]

        return centroids

    def _assign_labels(self, X):
        """
        Assigns each data point to the closest centroid.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: Array of labels for each data point.
        """

        # Distances between each example (data point) and every centroid.
        # distannces is shape (examples x number of centroids)
        distances = np.linalg.norm(
            X[:, np.newaxis, :] - self.centroids, ord=2, axis=2)

        # Assign each data point to the closest centroid by assigning
        # the centroid index as a label (axis = 1 says to take the min
        # for each row over the centroids)
        labels = np.argmin(distances, axis=1)

        return labels

    def _update_centroids(self, X, labels):
        """
        Updates the centroids based on the mean of the data points in each cluster.

        Args:
            X (np.ndarray): the input data.
            labels (np.ndarray): The current labels of the data points.

        Returns:
            np.ndarray: Updated centroids.
        """

        # Initialize new centroids to be zeros of the same shape
        new_centroids = np.zeros_like(self.centroids)

        for iCentroid in range(self.k):

            # All points assigned to centroid iCentroid
            cluster_points = X[labels == iCentroid]
            if len(cluster_points) > 0:
                # Assign new centroid to be the mean of all assigned points
                # to this cluster
                new_centroids[iCentroid] = np.mean(cluster_points, axis=0)
            else:
                # Empty cluseter (no points were assigned), so just intialize
                # the new centroid through a random number following a uniform distribution

                # Number of examples
                num_examples = X.shape[0]

                # Assign new centroid
                new_centroids[iCentroid] = X[np.random.choice(num_examples)]

        return new_centroids

    def fit(self, X):
        """
        Fit the data using the kmeans algorithm using kmeans++ initialization

        Args:
            X (numpy.ndarray): The data points, shape (n_samples, n_features).
        """

        # Initialize centroids using kmeans++ initialization
        self.centroids = self.kmeans_pp_init(X)

        for iter in range(self.max_iter):

            # Store a copy of the centroids
            old_centroids = self.centroids.copy()

            # Assign data samples to closest centroids
            self.labels = self._assign_labels(X)

            # Assign new centroids
            self.centroids = self._update_centroids(X, self.labels)

            # Check for convergence
            centroid_diff = np.linalg.norm(old_centroids-self.centroids)
            if np.all(centroid_diff < self.tol):
                print('Kmeans++ converged: # of iterations: ', iter)
                break
