from .display import show_letter, show_wordsearch, show_image
from .wordsearch import Masks
import numpy as np


class Data(object):
    def __init__(self, raw_data):
        self._raw_data = raw_data
        self.wordsearch1 = Wordsearch(raw_data['test1'])
        self.wordsearch2 = Wordsearch(raw_data['test2'])

    def get_train_letter(self, index):
        return Letter(self._raw_data['train_data'][index, :], self._raw_data['train_labels'][index])

    def process_wordsearch(self):
        for i in range(10):
            pass


class Wordsearch(object):
    def __init__(self, raw_data):
        self._raw_data = raw_data
        self._classified = False
        self.letters = list(self._iter_extract_letters())

    def find_word_fits(self, word):
        masks = Masks(word)
        fits = masks.get_fits(self)
        return fits

    def get_letter_at(self, coords):
        (x, y) = coords
        return self.letters(15 * y + x)

    def get_classified_array(self):
        if not self._classified:
            raise ValueError("not yet classified")
        letterlist = [l.label for l in self.letters]
        return np.array(letterlist).reshape((15, 15))

    def classify(self, classifier):
        for l in self.letters:
            l.classify(classifier)
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
        return show_wordsearch(self._raw_data)


class Letter(object):
    def __init__(self, raw_data, label=None):
        self._raw_data = raw_data
        self.label = label

    def show(self):
        return show_letter(self._raw_data)

    def classify(self, classifier):
        int_label = classifier.classify(self._raw_data)
        # self.label = chr(64 + int_label)
        self.label = int_label
        return self.label

    def __repr__(self):
        return 'Letter(<data>, ' + repr(self.label) + ')'
