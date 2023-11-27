'''
    File containing utils functions
'''

import numpy as np

def euclidian_distance(p1, p2):
    '''
        Method to calculate the euclidian distance between two points

        PARAMS:
            p1  - point 1 (can be of n dimentions)
            p2  - point 2 (can be of n dimentions)
    '''

    if not isinstance(p1, (np.ndarray, np.generic)):
        p1 = np.array(p1)

    if not isinstance(p2, (np.ndarray, np.generic)):
        p2 = np.array(p2)

    return np.sum(np.sqrt((p1 - p2) ** 2))


def calc_accuracy(actuals, preds):
    '''
        Method to calculate the accuracy for cateogrical predictions

        PARAMS:
            actuals - actuas (classes)
            pred    - predicted labels

        RETURNS:
            acc - float
    '''

    acc = np.sum(preds == actuals) / len(actuals)

    return acc

def mse(preds, actuals):
    '''
        Method to calculate the mean squared error 

        PARAMS:
            actuals - actuas (classes)
            pred    - predicted values

        RETURNS:
            mse value 
    '''
    return np.sum((preds - actuals) ** 2) / len(preds)


