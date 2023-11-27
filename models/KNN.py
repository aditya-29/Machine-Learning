'''
    KNN is a classification algorithm
        - It doesn't have any training per se
'''

from tqdm import tqdm
import utils as U
from models.BaseModel import Model
from collections import Counter
import numpy as np

class KNN(Model):
    def __init__(self, k):
        self.k = k
        self.mp = {}


    def fit(self, X, y):
        '''
            Method to fit the training data into the model

            PARAMS:
                X   - Training Data
                y   - Training labels

            Returns:
                None
        '''
        for i, ele in tqdm(enumerate(X)):
            ele = tuple(ele)

            if ele not in self.mp:
                self.mp[ele] = y[i]
            else:
                print("Data point with multiple prediction labels found : ", ele)
        
        print("Training Completed")

    def _predict(self , x):
        '''
            Internal method to execute the prediction

            PARAMS:
                x - single data point

            RETURNS
                pred    - pedicted output
        '''

        closest_neighbours = dict(sorted(self.mp.items(), key = lambda ele: U.euclidian_distance(ele[0], x)))
        pred_labels = [closest_neighbours[i] for i in closest_neighbours.keys()][:self.k]
        pred = Counter(pred_labels).most_common(1)

        return pred[0][0]


    def predict(self, X):
        '''
            Method to predict the testing data X

            PARAMS:
                X - Testing data

            RETURNS:
                y_pred  - Predicted labels
        '''

        predicted_labels = [self._predict(x) for x in X]
        return predicted_labels
    





