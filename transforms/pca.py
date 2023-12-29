from transforms.BaseTransform import Transform
import numpy as np

class PCA(Transform):
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None


    def fit(self, x):
        # mean
        self.mean = np.mean(x)
        x = x - self.mean

        # covariance
        cov = np.cov(x.T)

        # eigenvectors and eigen values
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T

        # sort the eigen vectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # get first "k" eigen vectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, x):
        x = x - self.mean
        print(x.shape)
        print(self.components.shape)
        return np.dot(x, self.components.T)