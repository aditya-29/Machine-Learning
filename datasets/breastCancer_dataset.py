import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from datasets.BaseDataset import BaseDataset 

cmap = ListedColormap(["#FF0000", "00FF00", "0000FF"])


class BreastCancer:
    def __init__(self, test_size = 0.2, random_state = 1234):
        '''
            Method to initialize the dataset
        '''
        cancer = datasets.load_breast_cancer()
        self.X, self.y = cancer.data, cancer.target
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, 
                                                                                self.y,
                                                                                test_size=test_size,
                                                                                random_state=random_state)
        
    def get_data(self):
        '''
            Method to return the train test and split data
        '''
        return self.x_train, self.x_test, self.y_train, self.y_test
    
    def visualize(self):
        '''
            Method to perform basic visya
        '''
        print("Shape of X-Train : ", self.x_train.shape)
        print("Shape of y-Train : ", self.y_train.shape)
        print("Shape of X-Test : ", self.x_test.shape)
        print("Shape of y-Test : ", self.y_test.shape)

        # fig = plt.figure(figsize = (8, 6))
        plt.scatter(self.x_train[:,2], self.y_train, color="b", marker="o", s=30)
        plt.show()