from models.BaseModel import Model
import numpy as np
from tqdm import tqdm
import utils as U
import matplotlib.pyplot as plt

class LinearRegression(Model):
    def __init__(self, lr = 0.01, epochs = 10, verbose = True):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.verbose = verbose

    def fit(self, X, y):
        '''
            Method to train the linear regression mode

            PARAMS:
                X   - X Train values
                y   - Y train values

            RETURNS:
                None
        '''

        if not isinstance(X, (np.ndarray, np.generic)):
            X = np.array(X)

        n_samples, n_features = X.shape

        self.weights = np.ones(n_features)
        self.bias = 0

        losses = []

        for e in tqdm(range(self.epochs)):
            y_pred = np.dot(X, self.weights) + self.bias

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
        return y_pred

    
        


        
