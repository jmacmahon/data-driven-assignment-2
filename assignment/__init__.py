"""Module containing all code for COM3004 assignment 2.

The main entry point is the `wordsearch` function.
"""

import matplotlib.pyplot as plt
import logging

from .classify import WeightedKNearestNeighbour
from .dimensionality import PCAReducer, BorderTrimReducer, DropFirstNSelector
from .pipeline import Pipeline

from .data import Wordsearch

logging.basicConfig(level=logging.INFO)


def wordsearch(test, words, train_data, labels, reducers=None):
    """Solve a wordsearch and display the solution.

    :param test: The input image as a 450x450 numpy array
    :param words: A list of words to search for in the wordsearch
    :param train_data: A numpy array of n 30x30 letter images
    :param labels: A numpy array of n letter labels corresponding to the
        train_data
    :param reducers: (Optional) Override which reducers to use
    """
    if reducers is None:
        reducers = [BorderTrimReducer(0, 4, 0, 3),
                    PCAReducer(11),
                    DropFirstNSelector(1)]
    pipeline = Pipeline(classifier=WeightedKNearestNeighbour(k=1, fuzzy=True),
                        reducers=reducers)
    pipeline.train(train_data, labels)

    wordsearch = Wordsearch(test, words)
    wordsearch.classify(pipeline)

    plt.figure(figsize=(12, 8))
    plt.axis('off')
    plt.subplot(1, 2, 1)
    wordsearch.show()

    plt.subplot(1, 2, 2)
    wordsearch.show_solved()

    plt.tight_layout()
    plt.show()


def load_and_wordsearch():
    """Helper function for solving the sample wordsearches provided."""
    from .load import load_data
    data = load_data()
    wordsearch(data.wordsearch1._raw_data, data.wordsearch1._words,
               data._raw_data['train_data'], data._raw_data['train_labels'],
               reducers=[])
    wordsearch(data.wordsearch1._raw_data, data.wordsearch1._words,
               data._raw_data['train_data'], data._raw_data['train_labels'])
    wordsearch(data.wordsearch2._raw_data, data.wordsearch2._words,
               data._raw_data['train_data'], data._raw_data['train_labels'])
