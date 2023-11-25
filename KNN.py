'''
    KNN is a classification algorithm
        - It doesn't have any training per se
'''

from tqdm import tqdm

class KNN:
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
        closest_neighbours = sorted(self.mp.items(), key = lambda x: )
        for x, y in self.mp.keys():


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

