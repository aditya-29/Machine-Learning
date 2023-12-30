# from models.BaseModel import Model
from typing import Any
import numpy as np
import utils as U
from models.BaseModel import Model
import matplotlib.pyplot as plt


class KMeans():
    def __init__(self, k=5, max_iters=1000, plot_steps=False):
        self.k=k
        self.max_iters=max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.k)]
        
        # mean feature vector for each cluster
        self.centroid = []
        self.alpha = 0.1

    def _create_clusters(self, centroids):
        '''
            Method function to create clusters based on the nearest centroids
        '''
        clusters = [[] for _ in range(self.k)]
        for idx, xi in enumerate(self.x):
            dist = [U.euclidian_distance(xi, ci) for ci in centroids]
            closest_label = np.argmin(dist)
            clusters[closest_label].append(idx)
        return clusters
    

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))

        for cluster_idx, cluster in enumerate(clusters):
            mean = np.mean(self.x[cluster], axis=0)
            centroids[cluster_idx] = mean
        return centroids
    
    def _is_converged(self, centroid_old):
        distances = [U.euclidian_distance(centroid_old[idx], self.centroids[idx]) for idx in range(self.k)]
        return sum(distances) == 0
    
    def _get_cluster_labels(self):
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels
            
    def predict(self, x):
        self.x = x
        self.n_samples, self.n_features = x.shape

        # initialize the centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [x[idx] for idx in random_sample_idxs]

        # optimization
        for _ in range(self.max_iters):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)

            # update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self.plot_steps:
                self.plot()

            # check if converged
            if self._is_converged(centroids_old):
                break
            
        # return cluster labels
        return self._get_cluster_labels()
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(12,8))

        for i, index in enumerate(self.clusters):
            point = self.x[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidths=2)

        plt.show()


            

