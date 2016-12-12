import numpy as np
from scipy.linalg import eigh

class PCAReducer(object):
    def __init__(self, train_data, n=40):
        self._train_data = train_data
        self._n = n

        self._train()

    def _train(self):
        cov = np.cov(self._train_data, rowvar=0)
        dim = cov.shape[0]
        _, eigenvectors = eigh(cov, eigvals=(dim - self._n, dim - 1))
        eigenvectors = np.fliplr(eigenvectors)
        self._eigenvectors = eigenvectors
