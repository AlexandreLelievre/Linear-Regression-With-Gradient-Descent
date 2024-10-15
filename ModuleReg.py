import numpy as np 
import matplotlib.pyplot as plt


class LinearReg():
    """
    Algorithme de descente de gradient pour la régression linéaire
    
    Parameters
    ----------
    features : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    alpha : float, optional (default=0.0003)
        Learning rate.
    epsilon: float, optional (default=1e-7)
            The tolerance for the test to stop the training process. 
    lam : float, optional (default=0)
        Regularization term
    l1_ratio : float, optional (default=0)
        L1/L2 regularization ratio
    
    Attributes
    ----------
    x_norm : numpy array
            Normalized value of x.
    x_denorm : numpy array
            Denormalized value of x.
    param : numpy array
        Regression parameters.
    yhat : numpy array
        Predicted values.
    costFct : float
        Cost function value.
    grads : numpy array
        Gradients of the cost function.
    costFctEvol : list
        Evolution of the cost function.
    y_denorm: float or numpy array
            The predicted denormalize value(s).
        
    Methods
    -------
    normalization(self, x)
        Normalize a variable x.
    denormalization(self,x_norm,x)
        Denormalize a variable x.
    hypothesis()
        Calculates the predicted values yhat using the current parameters.
    costFunction()
        Calculates the cost function using the current predicted values yhat and the true values y.
    gradients()
        Calculates the gradients of the cost function with respect to the parameters.
    updateParameters()
        Updates the parameters using the gradients and the learning rate.
    testCostFct(epsilon, prevCostFct)
        Tests whether the difference between the current and previous cost function is small enough.
    fit(epsilon=1e-9)
        Trains the model on the given data using gradient descent.
    plot()
        Plots the evolution of the cost function during the training process.
    predict(x)
        Makes a prediction for a given input x using the trained model.
    """
    def __init__(self, features, y, alpha = 0.0003, epsilon = 1e-7, lam = 0, l1_ratio = 0):
        self.lam = lam
        self.l1_ratio = l1_ratio
        self.features_base = features
        self.features =  features 
        
        self.m = self.features.shape[0]
        self.y_base = y
        self.y = self.normalization(y)
        self.epsilon = epsilon
        n = 2
        self.param = np.random.rand(n,1)
        self.yhat = self.hypothesis()
        self.alpha = alpha
        
    def normalization(self, x): 
        """
        Normalize a variable x.

        Parameters
        ----------
        x : numpy array
           Variable to normalize.

        Returns
        -------
        x_norm : numpy array
           Normalized value of x.
        """
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x_norm
        
    
    def denormalization(self,x_norm,x):
        """
        Denormalize a variable x.

        Parameters
        ----------
        x_norm : numpy array
            Normalized value of x.
        x : numpy array
            Original variable.

        Returns
        -------
        x_denorm : numpy array
            Denormalized value of x.
        """
        x_denorm = ((np.max(x)-np.min(x)) * x_norm) + np.min(x)
        return x_denorm
        
    
         
    def hypothesis(self):
        
        """
        Calculates the predicted values yhat using the current parameters.
        
        Returns
        -------
        numpy array
            The predicted values yhat.
        """
        
        self.yhat = (self.param[0]*self.features) + self.param[1]
        return self.yhat
        
    def costFunction(self):
        """
        Calculates the cost function using the current predicted values yhat and the true values y.
        
        Returns
        -------
        float
            The cost function.
        """
        regularisation = self.lam * ((1 - self.l1_ratio) * sum(abs(self.param[0])) + (self.l1_ratio * sum(self.param[0]**2)))
        
        self.costFct = (np.square(self.yhat - self.y).sum())/(2*self.m) + regularisation
        
        return self.costFct
    
    
    def gradients(self):
        """
        Calculate the gradients of the cost function for each parameter.
        
        Returns
        -------
        grads: numpy array
            The gradients of the cost function for each parameter.
        """
        
        dw = (2 * np.sum(self.features * (self.yhat - self.y)))/(2*self.m) + \
             self.lam * (1 - self.l1_ratio) * np.sign(self.param[0]) + 2 * self.l1_ratio * self.param[0]
        
        db = (2*np.sum(self.yhat-self.y))/(2*self.m) 
        
        self.grads = np.row_stack([dw, db])
        return self.grads

    def updateParameters(self):
        """
        Update the parameters using the gradients and the learning rate.
        
        Returns
        -------
        param: numpy array
        The updated parameters.
        """
        self.param = self.param - self.alpha * self.grads
        return self.param
    
    def testCostFct(self, prevCostFct):
        """
        Test if the difference between the current cost function and the previous cost function is small enough.
        
        Parameters
        ----------
        
        prevCostFct: float
        The value of the previous cost function.
        
        Returns
        -------
        bool
        True if the difference between the current and previous cost functions is greater than or equal to `epsilon` times the previous cost function, False otherwise.
        """
        a = self.costFunction()
        return np.abs(a - prevCostFct) >= self.epsilon* prevCostFct

    def fit(self):
        """
        Train the model on the given data by iteratively updating the parameters using the gradients of the cost function.
        
        
        Returns
        -------
        param: numpy array
            The trained parameters.
        """
        prevCostFct = 0
        self.costFctEvol = []
        count = 0
        
        while (self.testCostFct(prevCostFct)) &  (count < 100000):
            count += 1
            self.costFct = self.costFunction()
            self.grads = self.gradients()
            self.param = self.updateParameters()
            self.yhat = self.hypothesis()
            self.costFctEvol.append(self.costFct)
            prevCostFct = self.costFctEvol[-1]
        print("\nFinish: {} steps, cost function = {}".format(count, self.costFct))

        
        return self.param
    
    def plot(self):
        """
        Plot the evolution of the cost function during the training process.
        """
        return plt.plot(self.costFctEvol)
    
    def predict(self, x):
        """
        Predict the value for a given input using the trained parameters.
        
        Parameters
        ----------
        x: float or numpy array
            The input(s) to make a prediction for.
        
        Returns
        -------
        y_denorm: float or numpy array
            The predicted denormalize value(s).
        """
        
        
        y_norm = self.param[0]*x + self.param[1]
        y_denorm = self.denormalization(y_norm,self.y_base)
        return y_denorm