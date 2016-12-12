from .display import show_letter, show_wordsearch, show_image, letter_line
from .wordsearch import Masks
import numpy as np
import matplotlib.pyplot as plt


class Data(object):
    def __init__(self, raw_data):
        self._raw_data = raw_data
        self.wordsearch1 = Wordsearch(raw_data['test1'], raw_data['words'],
                                      raw_data['correct_coords'])
        self.wordsearch2 = Wordsearch(raw_data['test2'], raw_data['words'],
                                      raw_data['correct_coords'])

    def get_train_letter(self, index):
        return Letter(self._raw_data['train_data'][index, :],
                      self._raw_data['train_labels'][index])


class Wordsearch(object):
    def __init__(self, raw_data, words, solutions):
        self._raw_data = raw_data
        self._classified = False
        self.letters = list(self._iter_extract_letters())
        self._words = words
        self._solutions = solutions

    def find_line_image(self, word, rad=2):
        best = self.find_word_fits(word)[0]
        x, y = best['coords']
        x += 0.5
        y += 0.5
        return letter_line(x, y, best['direction'], len(word), rad=rad)

    # TODO refactor this into display
    def find_all_and_show(self, rad=2):
        lines = [self.find_line_image(word, rad=rad) for word in self._words]
        self.show()
        [plt.imshow(line, alpha=0.5) for line in lines]

    def find_all_coords(self):
        best_fits = [self.find_word_fits(word)[0] for word in self._words]
        def to_coords(fit):
            word = fit['word']
            start = fit['coords']
            end = (start[0] + len(word) * fit['direction'][0] - 1,
                   start[1] + len(word) * fit['direction'][1] - 1)
            return (word, (start, end))
        coords = dict([to_coords(fit) for fit in best_fits])
        return coords

    def correctness_score(self, only_score=True):
        num_correct = 0
        guesses = self.find_all_coords()
        incorrect_words = []
        for (word, solution_coords) in self._solutions.items():
            guess_coords = guesses[word]
            if solution_coords == guess_coords:
                num_correct += 1
            else:
                incorrect_words.append((word, (solution_coords, guess_coords)))
        if only_score:
            return num_correct
        else:
            return num_correct, dict(incorrect_words)

    def find_word_fits(self, word):
        masks = Masks(word)
        fits = masks.get_fits(self)
        return fits

    def get_letter_at(self, coords):
        (x, y) = coords
        return self.letters[15 * y + x]

    def get_classified_array(self):
        if not self._classified:
            raise ValueError("not yet classified")
        letterlist = [l.label_probabilities for l in self.letters]
        return np.array(letterlist).reshape(15, 15, 26)

    def classify(self, classifier):
        for l in self.letters:
            # TODO profile this -- could optimise as np multiplication?
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
    def __init__(self, raw_data, label_probabilities=None):
        self._raw_data = raw_data
        self.label_probabilities = label_probabilities

    def show(self):
        return show_letter(self._raw_data)

    def classify(self, classifier):
        self.label_probabilities = classifier.weighted_classify(self._raw_data)
        return self.label_probabilities

    def __repr__(self):
        label = np.argmax(self.label_probabilities) + 1
        return 'Letter(<data>, ' + repr(label) + ')'
