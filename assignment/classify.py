import numpy as np

class NearestNeighbour(object):
    def __init__(self, training_data, labels):
        self._training_data = training_data
        self._labels = labels

        self._modtrain = np.sqrt(np.sum(training_data ** 2, axis=1))

    def _get_cosine_distance(self, test, modtest, train_index):
        modtrain = self._modtrain[train_index]
        train = self._training_data[train_index]

        return np.dot(test, train)/(modtest * modtrain)

    def classify(self, test):
        distances = []
        modtest = np.sqrt(np.sum(test ** 2))
        for i in range(self._training_data.shape[0]):
            distances.append((self._get_cosine_distance(test, modtest, i), self._labels[i]))
        return max(distances)[1]
