import numpy as np
from itertools import chain

_centre_right = (1, 0)
_bottom_right = (1, 1)
_bottom_centre = (0, 1)
_bottom_left = (-1, 1)
_centre_left = (-1, 0)
_top_left = (-1, -1)
_top_centre = (0, -1)
_top_right = (1, -1)
_directions = [
    _centre_right, _bottom_right, _bottom_centre, _bottom_left, _centre_left,
    _top_left, _top_centre, _top_right
]


def gen_mask_direction(word, direction):
    word_numbers = [ord(c) - 96 for c in word.lower()]
    for y in range(15):
        for x in range(15):
            try:
                current = np.zeros((15, 15))
                xpos, ypos = x, y
                for i in word_numbers:
                    current[ypos, xpos] = i
                    xpos, ypos = (xpos + direction[0], ypos + direction[1])
                    if xpos < 0 or ypos < 0:
                        raise IndexError
            except IndexError:
                continue
            yield (current, x, y, direction)


def gen_mask(word):
    return chain.from_iterable(
        [gen_mask_direction(word, mut) for mut in _directions])


class Masks(object):
    def __init__(self, word):
        self._masks, self._xs, self._ys, self._directions = zip(
            *gen_mask(word))
        self._word = word

    def get_fits(self, wordsearch):
        # TODO instead of == here make it more fuzzy
        matches = self._masks == wordsearch.get_classified_array()
        sums = np.sum(matches, axis=(1, 2))
        # If this is too slow, look at np.argpartition
        best = np.argsort(sums)
        return MaskFits(sums, self._masks, self._xs, self._ys,
                        self._directions, best, self._word)

    def __getitem__(self, index):
        return {'mask': self._masks[index],
                'coords': (self._xs[index], self._ys[index]),
                'direction': self._directions[index]}


class MaskFits(object):
    def __init__(self, sums, masks, xs, ys, directions, best, word):
        self._sums = sums
        self._masks = masks
        self._xs = xs
        self._ys = ys
        self._directions = directions
        self._best = best
        self._word = word

    def __getitem__(self, index):
        orderedIndex = self._best[-(index + 1)]
        return {'mask': self._masks[orderedIndex],
                'coords': (self._xs[orderedIndex], self._ys[orderedIndex]),
                'direction': self._directions[orderedIndex],
                'score': self._sums[orderedIndex] / len(self._word)}
