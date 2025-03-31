import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


class NNRegression:

    def __init__(self, X, y, radius=1e-16):
        """
        Initializes the nearest neibghbor (NN) regression object.

        Parameters:
            X: numpy array, the input examples and features (m x n)
            y: numpy array, the target values (m x 1)
            raidus: float, the distance in which all nearest neighbor's
                       target values are averaged to produce an output.
        """

        # Averaging radial distance
        self.radius = radius

        # Store targets
        self.targets = y

        # Store KD tree
        self.KDTree = KDTree(X)

    def predict(self, X):
        """
        Predicts the target values for new input features.

        Parameters:
            X: numpy array, the input examples and features (m x n).

        Returns:
            numpy array, the predicted target values (m x 1).
        """

        # Initialize predictions array
        predictions = np.zeros((X.shape[0], 1))

        # Perform nearest neighbor search
        nn_list = self.KDTree.query_ball_point(X, self.radius, workers=-1)

        for i in range(X.shape[0]):

            # List of nearest neighbors
            nn = nn_list[i]

            # Set output
            number_nn = len(nn)

            if (number_nn == 0):
                # Search for closest point
                dist, nn = self.KDTree.query(X[i, :], eps=1e-8, workers=-1)
                nn = [nn]
                fact = 1.0
            else:
                # Averaging factor
                fact = 1.0/float(number_nn)

            # Perform averaging to generate output
            output = 0.0
            for iNN in nn:
                output += self.targets[iNN]

            predictions[i] = fact * output

        return predictions
