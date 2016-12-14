import matplotlib.pyplot as plt
import logging

from .classify import WeightedKNearestNeighbour
from .dimensionality import PCAReducer, BorderTrimReducer, DropFirstNSelector
from .pipeline import Pipeline

from .data import Wordsearch

logging.basicConfig(level=logging.INFO)


def wordsearch(test, words, train_data, labels):
    pipeline = Pipeline(classifier=WeightedKNearestNeighbour(k=1, fuzzy=True),
                        reducers=[BorderTrimReducer(0, 4, 0, 3),
                                  PCAReducer(11),
                                  DropFirstNSelector(1)])
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
    from .load import load_data
    data = load_data()
    wordsearch(data.wordsearch1._raw_data, data.wordsearch1._words,
               data._raw_data['train_data'], data._raw_data['train_labels'])
    wordsearch(data.wordsearch2._raw_data, data.wordsearch2._words,
               data._raw_data['train_data'], data._raw_data['train_labels'])
