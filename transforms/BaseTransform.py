from abc import ABC, abstractmethod

class Transform(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError("The fit() method is not implemeted.")

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError("The transform() method is not implemented.")
