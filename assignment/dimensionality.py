import numpy as np
from scipy.linalg import eigh
from scipy.stats import f_oneway
from logging import getLogger


class IdentityReducer(object):
    def train(self, *_, **__):
        pass

    def reduce(self, data, *_, **__):
        return data


class PCAReducer(IdentityReducer):
    def __init__(self, n=40):
        self._n = n

    def train(self, train_data, *_, **__):
        cov = np.cov(train_data, rowvar=0)
        dim = cov.shape[0]
        _, eigenvectors = eigh(cov, eigvals=(dim - self._n, dim - 1))
        eigenvectors = np.fliplr(eigenvectors)
        self._eigenvectors = eigenvectors

        self._mean = np.mean(train_data, axis=0)
        getLogger('assignment.dimensionality.pca')\
            .info("Trained PCA reducer ({} -> {} dimensions)"
                  .format(dim, self._n))

    def reduce(self, data, n=None):
        if n is None:
            n = self._n
        centred_data = data - self._mean
        vs = self._eigenvectors[:, :n]
        getLogger('assignment.dimensionality.pca')\
            .debug("PCA-reduced some samples")
        return np.dot(centred_data, vs)


class DropFirstNSelector(IdentityReducer):
    def __init__(self, n=1):
        self._n = n
        start_dim = "k"
        end_dim = "(k-{})".format(n)
        getLogger('assignment.dimensionality.dropfirstn')\
            .info("Init Drop First N feature selector ({} -> {} dimensions)"
                  .format(start_dim, end_dim))

    def reduce(self, data):
        return data.transpose()[self._n:].transpose()


class BestKSelector(IdentityReducer):
    def __init__(self, k):
        self._k = k

    def train(self, train_data, labels):
        classes = [train_data[labels == label] for label in np.unique(labels)]
        # Use f_oneway from scipy as a divergence measure
        scores = f_oneway(*classes).statistic
        self._best_k = np.argsort(scores)[::-1][:self._k]
        getLogger('assignment.dimensionality.bestk')\
            .info("Trained Best-K feature selector")

    def reduce(self, data):
        getLogger('assignment.dimensionality.bestk')\
            .debug("Best-K reduced some samples")
        return data.transpose()[self._best_k].transpose()


class BorderTrimReducer(IdentityReducer):
    def __init__(self, top=10, bottom=10, left=10, right=10,
                 startshape=(30, 30)):
        self._top = top
        self._bottom = bottom
        self._right = right
        self._left = left
        self._startshape = startshape
        start_dim = startshape[0] * startshape[1]
        end_dim = ((startshape[0] - top - bottom) *
                   (startshape[1] - left - right))
        getLogger('assignment.dimensionality.bordertrim')\
            .info("Init Border Trim Reducer ({} -> {} dimensions)"
                  .format(start_dim, end_dim))

    def reduce(self, data, flatten=True):
        data = data.reshape(-1, *self._startshape)
        height = self._startshape[0]
        width = self._startshape[1]
        out = data[:,
                   self._left:(width - self._right),
                   self._top:(height - self._bottom)]
        if not flatten:
            return out
        if data.shape[0] == 1:
            return out.flatten()
        new_len = ((width - self._left - self._right) *
                   (height - self._top - self._bottom))
        getLogger('assignment.dimensionality.bordertrim')\
            .debug("Trimmed border")
        return out.reshape(-1, new_len)
