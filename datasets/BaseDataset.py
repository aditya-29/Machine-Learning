import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod

class Dataset(ABC):
    @abstractmethod
    def get_data(self):
        '''
            Method to return the train test and split data.
        '''
        raise NotImplementedError("The method get_data() is not implemented")

    @abstractmethod
    def visualize(self):
        '''
            Method to perform basic visya
        '''
        raise NotImplementedError("The method visualize() is not implemented")
