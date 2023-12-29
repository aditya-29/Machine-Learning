from models.BaseModel import Model
import numpy as np

class SVM(Model):
    def __init__(self, learning_rate = 0.001, lambda_param=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = None

        
    def fit(self, x, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_feat = x.shape

        self.w = np.zeros(n_feat)
        self.b = 0

        for _ in range(self.epochs):
            for idx, xi in enumerate(x):
                condition = y_[idx] * (np.dot(xi, self.w) - self.b) >= 1

                if condition:
                    dw = 2*self.lambda_param*self.w
                    db = 0
                else:
                    dw = 2*self.lambda_param*self.w - (y_[idx]*xi)
                    db = y_[idx]

                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db


    def predict(self, x):
        linear = np.dot(x, self.w) - self.b
        return np.sign(linear)
