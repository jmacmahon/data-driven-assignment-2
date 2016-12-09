import numpy as np
from scipy.stats import mode


class Classifier(object):
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


class KNearestNeighbour(Classifier):
    def __init__(self, training_data, labels, k=1):
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
