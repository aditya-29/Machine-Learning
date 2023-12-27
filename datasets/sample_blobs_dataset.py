import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from datasets.BaseDataset import Dataset 

class SampleBlobs(Dataset):
    def __init__(self, n_samples = 150, 
                 n_features = 2, 
                 centers = 2,
                 cluster_std = 1.05, 
                 random_state = 1234,
                 test_size=0.2):
        '''
            Method to initialize the dataset
        '''

        X, y = datasets.make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=random_state)
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
        fig.add_subplot(1,1,1)
        plt.scatter(self.x_train[:,0], self.x_train[:,1], marker="o", s=30, c=self.y_train)
        plt.show()  