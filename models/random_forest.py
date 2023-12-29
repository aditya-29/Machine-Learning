from models.BaseModel import Model
from models import DecisionTree
import numpy as np
from collections import Counter
from tqdm import tqdm

def bootstrap_sample(x, y):
    n_samples = x.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return x[idxs,:], y[idxs]

def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common

class RandomForest(Model):
    def __init__(self, n_trees=100, n_feats=None, max_depth=100, min_samples_split=2):
        self.n_trees=n_trees
        self.n_feats = n_feats
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, x, y):
        self.trees = []

        for _ in tqdm(range(self.n_trees)):
            tree = DecisionTree(n_feats = self.n_feats, max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            x_sample, y_sample = bootstrap_sample(x, y)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)

    def predict(self, x):
        tree_preds = np.array([tree.predict(x) for tree in self.trees])
        # rearange the dimentions
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        max_labels = [most_common_label(pred) for pred in tree_preds]
        return np.array(max_labels)