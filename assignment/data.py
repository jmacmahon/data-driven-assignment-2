"""Classes which hold data about various objects in the system."""

from .display import show_letter, show_wordsearch, show_image, draw_lines
from .wordsearch import Masks
import numpy as np
import matplotlib.pyplot as plt
from logging import getLogger


class Data(object):
    """Convenience wrapper for the data provided with the assignment.

    :param raw_data: The data contained in the Pickle file
    """

    def __init__(self, raw_data):
        """See class docstring."""
        self._raw_data = raw_data
        self.wordsearch1 = Wordsearch(raw_data['test1'], raw_data['words'],
                                      raw_data['correct_coords'])
        self.wordsearch2 = Wordsearch(raw_data['test2'], raw_data['words'],
                                      raw_data['correct_coords'])

    def get_train_letter(self, index):
        """Get a particular letter and its label from the training data."""
        return Letter(self._raw_data['train_data'][index, :],
                      self._raw_data['train_labels'][index])


class Wordsearch(object):
    """Hold data about a wordsearch and perform operations on it.

    :param raw_data: The 450x450 numpy array containing the wordsearch image
    :param words: The list of words to search for
    :param solutions: (Optional) The correct solutions to score against, in the
        form of a dictionary: `{'word': ((startx, starty), (endx, endy))}`
    """

    def __init__(self, raw_data, words, solutions=None):
        """See class docstring."""
        self._raw_data = raw_data
        self._classified = False
        self.letters = list(self._iter_extract_letters())
        self._words = words
        self._solutions = solutions
        self._coords = None

    @property
    def coords(self):
        """The coordinates of the words in the solved wordsearch.

        :note: On first access it runs all the wordsearch-solving code, and
            thereafter returns a cached value.
        """
        if self._coords is None:
            self._coords = self._find_all_coords()
        return self._coords

    def _find_all_coords(self):
        best_fits = [self._find_word_fits(word)[0] for word in self._words]

        def to_coords(fit):
            word = fit['word']
            start = fit['coords']
            direction = fit['direction']
            end = (start[0] + len(word) * fit['direction'][0] - 1,
                   start[1] + len(word) * fit['direction'][1] - 1)
            return (word, (start, end, direction))

        coords = dict([to_coords(fit) for fit in best_fits])
        getLogger('assignment.data.wordsearch')\
            .info("Found best fits for all words")
        return coords

    def correctness_score(self, only_score=True):
        """Compare the solved solutions with the reference solutions provided.

        :param only_score: (Default = True) When set to false, some data about
            the incorrect words is also returned.

        :return: The number of correctly-solved words and optionally a `dict`
            of incorrect words and their guessed and true coordinates.

        :raises ValueError: If no solutions were provided.
        """
        if self._solutions is None:
            raise ValueError("No solutions provided")
        num_correct = 0
        incorrect_words = []
        for (word, solution_coords) in self._solutions.items():
            guess_coords = self.coords[word][0:2]
            if solution_coords == guess_coords:
                num_correct += 1
            else:
                incorrect_words.append((word, (solution_coords, guess_coords)))
        if only_score:
            return num_correct
        else:
            return num_correct, dict(incorrect_words)

    def _find_word_fits(self, word):
        masks = Masks(word)
        fits = masks.get_fits(self)
        return fits

    def get_letter_at(self, coords):
        """Get the Letter object at the specified coordinates."""
        (x, y) = coords
        return self.letters[15 * y + x]

    def get_classified_array(self):
        """Get the classification probability distribution for each letter.

        :return: A 15x15x26 array of the probabilities for each letter, 26
            being the number of labels in the probability distribution.

        :raises ValueError: If the letters have not yet beein classified.
        """
        if not self._classified:
            raise ValueError("not yet classified")
        letterlist = [l.label_probabilities for l in self.letters]
        return np.array(letterlist).reshape(15, 15, 26)

    def classify(self, pipeline):
        """Iteratively classify each letter in the wordsearch."""
        for l in self.letters:
            # TODO profile this -- could optimise as np multiplication?
            l.classify(pipeline)
        getLogger('assignment.data.wordsearch')\
            .info("Classified all letters")
        self._classified = True

    def _iter_extract_letters(self):
        for y in range(0, 15):
            for x in range(0, 15):
                yield self._extract_letter_at((x, y))

    def _extract_letter_at(self, coords):
        (x, y) = coords
        xmin = 30*x
        xmax = 30*(x+1)
        ymin = 30*y
        ymax = 30*(y+1)
        pixels = []
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                pixels.append((self._raw_data[y, x]))
        return Letter(np.array(pixels))

    def show(self):
        """Display (using pyplot) the plain un-annotated wordsearch."""
        return show_wordsearch(self._raw_data)

    def show_solved(self, rad=2):
        """Display (using pyplot) the wordsearch annotated with solutions."""
        self.show()
        return draw_lines(self.coords, rad=rad)


class Letter(object):
    """Hold data about an individual letter within the wordsearch.

    :param raw_data: The 30x30 image of the letter.
    :param label_probabilities: The probability distribution of the letter's
        class label.
    """

    def __init__(self, raw_data, label_probabilities=None):
        """See class docstring."""
        self._raw_data = raw_data
        self.label_probabilities = label_probabilities

    def show(self):
        """Display the letter image using pyplot."""
        return show_letter(self._raw_data)

    def classify(self, pipeline):
        """Classify the letter with the provided classification pipeline.

        :param pipeline: The classification pipeline to process the image data
        """
        self.label_probabilities = pipeline(self._raw_data)
        return self.label_probabilities

    def __repr__(self):
        """Prettier string representation of a letter."""
        label = np.argmax(self.label_probabilities) + 1
        return 'Letter(<data>, ' + repr(label) + ')'
