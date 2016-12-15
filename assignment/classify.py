"""Contains all the classifier code."""

import numpy as np
from scipy.stats import mode
from logging import getLogger


class Classifier(object):
    """Abstract superclass containing classifier evaluation code."""

    def __init__(self):
        """Not to be instantiated directly.

        :note: When subclassing, put any arguments to your `__init__` in
            `self._kwargs`.
        """
        self._confusion_matrix = None
        self._normalised_confusion_matrix = None

    def loo_test(self):
        """Perform a leave-one-out (LOO) test of the classifier.

        :return: A confusion matrix, a percentage correct and a number
            incorrect
        """
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
        getLogger('assignment.classifier').info("Performed LOO testing")
        return confusion_matrix, correct / float(n), (n - correct)

    def partition_test(self, n):
        """Perform a partition test of the classifier.

        :param n: The number of samples to use for testing

        :return: A percent correct and a number incorrect
        """
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
        getLogger('assignment.classifier').info("Performed partition testing")
        return correct / float(n), (n - correct)

    @property
    def confusion_matrix(self):
        """Get the LOO-testing-based confusion matrix of the classifier.

        :note: On first access it runs the LOO testing to establish the
            confusion matrix, and then caches the result for subsequent calls.

        :return: The confusion matrix
        """
        if self._confusion_matrix is None:
            self._confusion_matrix = self.loo_test()[0]
        getLogger('assignment.classifier').debug("Built confusion matrix")
        return self._confusion_matrix

    @property
    def normalised_confusion_matrix(self):
        """Get a normalised confusion matrix for the classifier.

        This is a confusion matrix with rows normalised such that the sum of
        every row is 1.  I.e. each row is a probability distribution of the
        actual label given the classified label.

        :return: The normalised confusion matrix
        """
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
            getLogger('assignment.classifier')\
                .info("Built normalised confusion matrix")
        return self._normalised_confusion_matrix


class KNearestNeighbour(Classifier):
    """A basic k-nearest-neighbour classifier.

    :param training_data: The n data points to train on
    :param labels: The n labels corresponding to the data points in
        training_data
    :param k: (Default = 1) The k in k-nearest-neighbour
    """

    def __init__(self, training_data, labels, k=1):
        """See class docstring."""
        super().__init__()
        self._training_data = training_data
        self._labels = labels
        self._k = k
        self._kwargs = {"k": k}

        self._modtrain = np.sqrt(np.sum(training_data ** 2, axis=1))

    def classify(self, test):
        """Classify a test data point.

        :return: The best-guess label
        """
        modtest = np.sqrt(np.sum(test ** 2))
        dots = np.dot(test, self._training_data.transpose())
        distances = dots/(modtest * self._modtrain)
        nearest_k_indices = np.argsort(distances)[::-1][0:self._k]
        best_index = mode(nearest_k_indices).mode
        if len(test.shape) == 1:
            samples = 1
            dim = test.shape[0]
        else:
            samples = test.shape[-1]
            dim = test.shape[:-1]
        getLogger('assignment.classifier.knn')\
            .debug("Classified {} samples of {} dimensions"
                   .format(samples, dim))
        return self._labels[best_index]


class WeightedKNearestNeighbour(Classifier):
    """A weighted/"fuzzy" k-nearest-neighbour classifier.

    Instead of classifying as a single label, this returns a probability
    distribution on the labels.

    :param k: The k in k-nearest-neighbour
    :param fuzzy: (Default = True) Whether to enable weighted/"fuzzy"
        classifying.  If set to false, this behaves like a normal
        k-nearest-neighbour classifier but returns its label as a probability
        distribution with a single possible label.
    """

    def __init__(self, k=1, fuzzy=True):
        """See class docstring."""
        super().__init__()
        self._k = k
        self._fuzzy = fuzzy

    def train(self, training_data, labels):
        """Train the classifier.

        :param training_data: The n data points to train on
        :param labels: The n labels corresponding to the data points in
            training_data
        """
        self._classifier = KNearestNeighbour(training_data, labels, self._k)
        self._classifier.normalised_confusion_matrix
        getLogger('assignment.classifier.weightedknn')\
            .info("Trained weighted k-NN classifier with fuzzy = {}"
                  .format(self._fuzzy))

    def classify(self, test):
        """Classify a test data point.

        :return: A probability distribution over the labels
        """
        # Fuzziness slightly improves accuracy on poor quality image
        best_label = self._classifier.classify(test)
        if self._fuzzy:
            confusion_matrix = self._classifier.normalised_confusion_matrix
            probabilities = confusion_matrix[best_label - 1]
        else:
            probabilities = np.zeros(26)
            probabilities[best_label - 1] = 1
        if len(test.shape) == 1:
            samples = 1
            dim = test.shape[0]
        else:
            samples = test.shape[-1]
            dim = test.shape[:-1]
        getLogger('assignment.classifier.weightedknn')\
            .debug("Classified {} samples of {} dimensions"
                   .format(samples, dim))
        return probabilities
