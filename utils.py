'''
    File containing utils functions
'''

import numpy as np
from graphviz import Digraph

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

def _trace(root):
    '''
        Method to build the nodes and edges by recursively calling them

        PARAMS:
            root    - start node
    '''

    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)

            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def visualize_micro_grad(root):
    '''
        Method to visualize the values built using micrograd

        PARAMS:
            root    - start node
    '''
    dot = Digraph(format = "svg", graph_attr = {"rankdir" : "LR"})

    nodes, edges = _trace(root)

    for n in nodes:
        uid = str(id(n))
        # for any value in the graph create a rectangular ("record") node for it
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad, ), shape = "record")

        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot