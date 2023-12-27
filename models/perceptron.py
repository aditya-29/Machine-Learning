import numpy as np

from models.BaseModel import Model
from activations import UnitStep

class Perceptron(Model):
    def __init__(self, lr=0.01, epochs = 1000, activation = UnitStep):
        self.lr = lr
        self.epochs = epochs
        self.activation = activation
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for e in range(self.epochs):
            for idx, xi in enumerate(X):
                linear_output = np.dot(xi, self.weights) + self.bias
                out = self.activation(linear_output)

                # update
                self.weights += self.lr * (y[idx] - out) * xi
                self.bias += self.lr * (y[idx] - out)
        

    def predict(self, X):
        f = np.dot(X, self.weights) + self.bias
        return self.activation(f)