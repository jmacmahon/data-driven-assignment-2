import numpy as np
from scipy.linalg import eigh

class PCAReducer(object):
    def __init__(self, n=40):
        self._n = n

    def train(self, train_data, *_, **__):
        cov = np.cov(train_data, rowvar=0)
        dim = cov.shape[0]
        _, eigenvectors = eigh(cov, eigvals=(dim - self._n, dim - 1))
        eigenvectors = np.fliplr(eigenvectors)
        self._eigenvectors = eigenvectors

        self._mean = np.mean(train_data, axis=0)

    def reduce(self, data, n=None):
        if n is None:
            n = self._n
        centred_data = data - self._mean
        vs = self._eigenvectors[:, :n]
        return np.dot(centred_data, vs)
