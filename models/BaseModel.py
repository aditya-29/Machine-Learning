from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError("The fit() method is not implemeted.")

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError("The predict() method is not implemented.")
