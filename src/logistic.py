import numpy as np
import math


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # define a sigmoid function
        def sigmoid(z):
            s = 1.0/(1.0+math.exp(-z))
            s = np.minimum(s, 0.999999999999999)  # Set upper bound
            s = np.maximum(s, 0.000000000000001)  # Set lower bound
            return s



        if self.theta is None:
            self.theta = np.zeros((x.shape[1],1)) # create a column zero vector
        iteration = 0 # keep track of current iterations
        diff = 10000 * self.eps # initialize to be infinity
        n = x.shape[0] # number of examples


        while diff >= self.eps:
            # re-initialize the Hessian and Gradient matrix to be 0
            H = np.zeros((x.shape[1], x.shape[1]))  # square matrix used to hold Hessian matrix
            G = np.zeros((x.shape[1], 1))  # for gradient

            # calculate the gradient of cost function
            for i in range(n):
                x_i = x[i,:].reshape(-1,1)
                G = G + (-1/n)*(y[i]-sigmoid(np.dot(self.theta.T,x_i)))*x_i

            # calculate the Hessian
            for k in range(H.shape[0]):
                for l in range(H.shape[1]):
                    for i in range(n):
                        x_i = x[i, :].reshape(-1, 1)
                        H[k,l] = H[k,l] + (1/n) * sigmoid(np.dot(self.theta.T,x_i)) * (1-sigmoid(np.dot(self.theta.T,x_i))) * x_i[k,0] * x_i[l,0]

            prev_theta = self.theta
            self.theta = self.theta - self.step_size*np.dot(np.linalg.inv(H),G)

            diff = np.sum(np.abs(self.theta-prev_theta))
            iteration += 1

            if self.verbose == True:
                # print loss values:
                loss = 0
                for i in range(n):
                    x_i = x[i, :].reshape(-1, 1)
                    loss += (-1/n) * (y[i]*math.log(sigmoid(np.dot(self.theta.T,x_i))) + (1-y[i])*math.log(1-sigmoid(np.dot(self.theta.T,x_i))))
                print(loss)


            if iteration >= self.max_iter:
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        def sigmoid(z):
            s = 1.0/(1.0+math.exp(-z))
            #s = np.minimum(s, 0.999999999999999)  # Set upper bound
            #s = np.maximum(s, 0.000000000000001)  # Set lower bound
            return s
        py = np.zeros(x.shape[0]) # 1d array

        for i in range(x.shape[0]):
            py[i] = sigmoid(np.dot(self.theta.T, x[i,:].reshape(-1,1)))


        return py

        # *** END CODE HERE ***