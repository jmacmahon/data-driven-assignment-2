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


def gen_word_pattern(word, direction):
    word_numbers = [ord(c) - 97 for c in word.lower()]
    word_pattern = np.zeros((
        abs(len(word) * direction[1]) or 1,
        abs(len(word) * direction[0]) or 1,
        26
    ))
    xpos = -1 if direction[0] < 0 else 0
    ypos = -1 if direction[1] < 0 else 0
    for i in word_numbers:
        word_pattern[ypos, xpos, i] = 1
        xpos += direction[0]
        ypos += direction[1]
    return word_pattern


def gen_mask_direction(word, direction):
    # Tried optimising with
    # - pre-generating a word pattern array and overlaying at different
    #   positions (fuzzy_stats_3)
    # - using a 5-D 'masks' array rather than recreating "current" each time
    #   (fuzzy_stats_4)
    word_numbers = [ord(c) - 97 for c in word.lower()]

    for y in range(15):
        for x in range(15):
            try:
                current = np.zeros((15, 15, 26))
                xpos, ypos = x, y
                for i in word_numbers:
                    current[ypos, xpos, i] = 1
                    xpos, ypos = (xpos + direction[0], ypos + direction[1])
                    if xpos < 0 or ypos < 0:
                        raise IndexError
            except (IndexError, ValueError):
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
        matches = np.array(self._masks) * wordsearch.get_classified_array()
        # matches = np.zeros((559, 15, 15, 26))
        sums = np.sum(matches, axis=(1, 2, 3))
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
