import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from datasets.BaseDataset import Dataset


class SampleClassification(Dataset):
    def __init__(self, n_samples = 1000, 
                 n_features = 10, 
                 n_classes = 2,
                 test_size = 0.2, 
                 random_state = 123):
        '''
            Method to initialize the dataset
        '''

        X, y = datasets.make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, random_state=random_state)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


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

        fig = plt.figure(figsize = (8,6))
        plt.scatter(self.x_train[:,0], self.y_train, color="b", marker="o", s=30)
        plt.show()