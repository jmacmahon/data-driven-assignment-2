import numpy as np
from scipy.stats import mode


class Classifier(object):
    def __init__(self):
        self._confusion_matrix = None
        self._normalised_confusion_matrix = None

    def loo_test(self):
        def loo_test_indiv(index):
            try:
                train_data = np.vstack((self._training_data[0:index, :],
                                        self._training_data[index + 1, :]))
                train_labels = np.hstack((self._labels[0:index],
                                          self._labels[index + 1]))
            except IndexError:
                train_data = self._training_data[0:index, :]
                train_labels = self._labels[0:index]
            test_data = self._training_data[index, :]
            test_label = self._labels[index]
            classifier = self.__class__(train_data, train_labels,
                                        **self._kwargs)
            return (classifier.classify(test_data), test_label)
        correct = 0
        n = self._training_data.shape[0]
        classes = np.unique(self._labels).shape[0]
        confusion_matrix = np.zeros((classes, classes))
        for i in range(n):
            classified_label, actual_label = loo_test_indiv(i)
            # Assume labels are 1-indexed
            confusion_matrix[classified_label - 1, actual_label - 1] += 1
            if classified_label == actual_label:
                correct += 1
        return confusion_matrix, correct / float(n), (n - correct)

    def partition_test(self, n):
        test_data = self._training_data[0:n, :]
        test_labels = self._labels[0:n]
        train_data = self._training_data[n:, :]
        train_labels = self._labels[n:]

        test_classifier = self.__class__(train_data, train_labels,
                                         **self._kwargs)
        # If this is too slow look at np.vectorize
        correct = 0
        for i in range(n):
            if test_labels[i] == test_classifier.classify(test_data[i, :]):
                correct += 1
        # TODO confusion matrix
        return correct / float(n), (n - correct)

    @property
    def confusion_matrix(self):
        if self._confusion_matrix is None:
            self._confusion_matrix = self.loo_test()[0]
        return self._confusion_matrix

    @property
    def normalised_confusion_matrix(self):
        if self._normalised_confusion_matrix is None:
            confusion_matrix = self.confusion_matrix
            length = confusion_matrix.shape[0]
            sums = np.sum(confusion_matrix, axis=1).reshape(length, 1)
            # Handle dividing by 0-sums
            with np.errstate(divide='ignore', invalid='ignore'):
                normalised_confusion_matrix = confusion_matrix/sums
            for i in range(normalised_confusion_matrix.shape[0]):
                if (~np.isfinite(normalised_confusion_matrix[i, i])):
                    # new_row = np.zeros(length)
                    # new_row[i] = 1
                    new_row = np.full(length, 1.0/length)
                    normalised_confusion_matrix[i] = new_row
            self._normalised_confusion_matrix = normalised_confusion_matrix
        return self._normalised_confusion_matrix


class KNearestNeighbour(Classifier):
    def __init__(self, training_data, labels, k=1):
        super().__init__()
        self._training_data = training_data
        self._labels = labels
        self._k = k
        self._kwargs = {"k": k}

        self._modtrain = np.sqrt(np.sum(training_data ** 2, axis=1))

    def classify(self, test):
        modtest = np.sqrt(np.sum(test ** 2))
        dots = np.dot(test, self._training_data.transpose())
        distances = dots/(modtest * self._modtrain)
        nearest_k_indices = np.argsort(distances)[::-1][0:self._k]
        best_index = mode(nearest_k_indices).mode
        return self._labels[best_index]


class WeightedKNearestNeighbour(Classifier):
    def __init__(self, k=1, fuzzy=True):
        super().__init__()
        self._k = k
        self._fuzzy = fuzzy

    def train(self, training_data, labels):
        self._classifier = KNearestNeighbour(training_data, labels, self._k)

    def classify(self, test):
        # Fuzziness slightly improves accuracy on poor quality image
        best_label = self._classifier.classify(test)
        if self._fuzzy:
            confusion_matrix = self._classifier.normalised_confusion_matrix
            probabilities = confusion_matrix[best_label - 1]
        else:
            probabilities = np.zeros(26)
            probabilities[best_label - 1] = 1
        return probabilities
