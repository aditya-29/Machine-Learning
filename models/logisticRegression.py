from models.BaseModel import Model
import numpy as np
import matplotlib.pyplot as plt
from activations import Sigmoid
from loss import Categorical_CrossEntropy
import utils as U

class LogisticRegression(Model):
    def __init__(self, lr = 0.01, epochs = 10, verbose = True):
        '''
            Method to initialize the Logitstic Regression Model
        '''
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose

        self.weights = None
        self.bias = None

    def fit(self, X, y):
        '''
            Method to train the Logistic Regression model

            PARAMS:
                X   - X train values
                y   - y values

            RETURNS:
                None
        '''

        n_samples, n_features = X.shape
        self.weights = np.ones(n_features)
        self.bias = 0   

        losses = []

        for e in range(self.epochs):
            f = np.dot(X, self.weights) + self.bias
            y_pred = Sigmoid(f)

            loss = U.mse(y_pred, y)
            losses.append(loss)

            dw = 2 * np.dot(X.T, y_pred - y) / len(y_pred)
            db = 2 * np.sum(y_pred - y) / len(y_pred)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        if self.verbose:
            plt.plot(range(self.epochs), losses)
            plt.xlabel("Number of Epochs")
            plt.ylabel("MSE")
            plt.title("Training Loss Graph")

        print("Training Completed")

    def predict(self, X):
        '''
            Method to predict the X_Test values

            PARAMS:
                X   - X test values

            RETURNS:
                y_pred  - predict y values
        '''
        y_pred = np.dot(X, self.weights) + self.bias
        y_pred = Sigmoid(y_pred)
        
        return y_pred


