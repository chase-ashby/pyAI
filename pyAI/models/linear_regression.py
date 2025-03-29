import numpy as np
import matplotlib.pyplot as plt

# Global constants
gradient_descent = 0
normal_equations = 1

class LinearRegression:
    """
    A class for performing serial linear regression using gradient descent.
    """

    def __init__(self, learning_rate=0.1, iterations=1000, tolerance=1e-8, method="gd"):
        """
        Initializes the SerialLinearRegression object.

        Parameters:
        learning_rate: float, the learning rate for gradient descent.
        iterations: int, the maximum number of iterations.
        tolerance: float, the tolerance for convergence.
        method: string, solving method.
                gd = gradient descent
                ne = normal equations
        """

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tolerance = tolerance
        self.cost_history = []

        if (method == "gd"):
            self.method = gradient_descent
        elif (method == "ne"):
            self.method = normal_equations
        else:
            err_msg = f"""
              Invalid linear regression method: {self.method}
              Valid choices: 
              - 'gd' = gradient descent
              - 'ne' = normal equations
              """
            raise ValueError(err_msg)
        
        # Weights including bias
        self.theta = None

    def _add_bias(self, X):
        """
        Add bias column to input data matrix.

        Parameters:
            X: numpy array, the input examples and features (m x n).

        Returns:
            numpy array, the data matrix with an added bias column (m x n+1)
        """

        # Number of examples
        m = X.shape[0]

        return np.hstack((X, np.ones((m, 1))))
    
    def _gradientMSE(self, X, y):
        """
        Compute gradient of the mean squared error (MSE)

        Parameters:
            X: numpy array, the input examples and features with bias (m x n+1)
            y: numpy array, the target values (m x 1).

        Returns: 
            numpy array, the gradient of the mean squared error (n+1 x 1)
        """

        m = X.shape[0]
        fact = 2.0/float(m)
        grad_mse = fact * (X.T @ X @ self.theta - X.T @ y)

        return grad_mse
    
    def _mse(self, Xb, y):
        """
        Compute the mean squared error (MSE) between predicted outputs
        and the correct outputs

        Parameters:
            Xb: numpy array, the input examples and features with bias (m x n + 1)
            y: numpy array, the target values (m x 1).

        Returns:
            float, the mean squared error 
        """

        # Predicted output by applying the model
        predicted_output = Xb @ self.theta

        # Multiplication factor for average
        m = Xb.shape[0]
        fact = 1.0/float(m)

        return fact * (np.linalg.norm(predicted_output - y)**2) 
    
    def _fit_gradient_descent(self, Xb, y):
        """
        Generate a linear regression model through gradient descent training.

        Parameters:
            Xb: numpy array, the input bias examples and features (m x n+1).
            y: numpy array, the target values (m x 1).
        """

        # Initialize weights to unity
        self.theta = np.ones((Xb.shape[1],1))

        # Initial cost
        self.cost_history.append(self._mse(Xb, y))

        # Perform gradient descent for a fixed number of 
        # iterations or until convergence with a fixed
        # learning rate (will be replaced by backtracking line search)
        for iter in range(self.iterations):

            # Compute gradient
            grad = self._gradientMSE(Xb, y)

            # Update weights
            self.theta = self.theta - self.learning_rate * grad

            # Cost function (MSE)
            cost = self._mse(Xb, y)

            # Add cost to history
            self.cost_history.append(cost)

            # Check for convergence
            if (cost < self.tolerance): break

        return None

    def _fit_normal_equations(self, Xb, y):
        """
        Generate a linear regression model through gradient descent training.

        Parameters:
            Xb: numpy array, the input bias examples and features (m x n+1).
            y: numpy array, the target values (m x 1).
        """
        
        # Initial weights
        self.theta = np.ones((Xb.shape[1], 1))
        
        try:
            self.theta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
        except np.linalg.LinAlgError:
            print("The normal equations may not have a unique solution.")

        return None
    
    def fit(self, X, y):
        """
        Generate a linear regression model using either gradient descent 
        or the normal equations.
            X: numpy array, data matrix of examples and features (m x n)
            y: numpy array, the target values (m x 1).
        """

        # Add bias column to input data
        Xb = self._add_bias(X)

        # Call solver method
        match self.method:
            case 0:
                print ("Performing fitting using gradient descent")
                self._fit_gradient_descent(Xb, y)
            case 1: 
                print ("Performing fitting using normal equations")
                self._fit_normal_equations(Xb, y)
            case _:
                raise ValueError("Invalid linear regression solver method.")
            
        return None

    def predict(self, X):
        """
        Predicts the target values for new input features.

        Parameters:
            X: numpy array, the input examples and features (m x n).

        Returns:
            numpy array, the predicted target values (m x 1).
        """

        # Add bias column to input data
        Xb = self._add_bias(X)

        return Xb @ self.theta
    
    def plot_model(self, X, y):
        """
        Visualize the linear regression model.

        Parameters:
            X: numpy array, the examples and features (m x n)
            y: numpy array, the target values (m x 1)
        """

        # Number of features 
        n = X.shape[1]

        if n == 1:
            print('here in plot n == 1')
            plt.scatter(X, y)
            plt.plot(X, self.predict(X), color='red')
            plt.xlabel('Feature')
            plt.ylabel('Target')
            plt.title('Linear Regression Fit')
        elif n == 2:
            xmin, xmax = X[:, 0].min()-0.1, X[:, 0].max()+0.1
            ymin, ymax = X[:, 1].min()-0.1, X[:, 1].max()+0.1
            xg, yg = np.meshgrid(np.arange(xmin, xmax, 0.01),
                                 np.arange(ymin, ymax, 0.01))
            Z = self.predict(np.c_[xg.ravel(), yg.ravel()]).reshape(xg.shape)
            plt.contourf(xg, yg, Z, levels=40, alpha=1.0)
            plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors='k')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Linear Regression Contour')
        else:
            raise Exception("Cannot plot models with 0 or more than 2 features.")
        plt.show()

    def plot_cost_history(self):
        plt.title('Cost History')
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        
        nIter = len(self.cost_history)
        iters = np.linspace(1,nIter,nIter)
        print(self.cost_history)
        plt.semilogy(iters,self.cost_history,'b')
        plt.show()
        