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

    if not p1.isintance(np.array):
        p1 = np.array(p1)

    if not p2.isinstance(np.array):
        p2 = np.array(p2)

    return np.sum(np.sqrt((p1 - p2) ** 2))
