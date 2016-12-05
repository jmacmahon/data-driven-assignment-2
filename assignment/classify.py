import numpy as np

class NearestNeighbour(object):
    def __init__(self, training_data, labels):
        self._training_data = training_data
        self._labels = labels

        self._modtrain = np.sqrt(np.sum(training_data ** 2, axis=1))

    def classify(self, test):
        modtest = np.sqrt(np.sum(test ** 2))
        dots = np.dot(test, self._training_data.transpose())
        distances = dots/(modtest * self._modtrain)
        nearest_index = np.argmax(distances)
        return self._labels[nearest_index]
