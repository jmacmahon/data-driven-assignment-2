"""All the code for solving a wordsearch based on pre-classified letters."""

import numpy as np
from itertools import chain
from logging import getLogger

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
    """Generator of word masks in a particular direction.

    :return: A generator which yields a 15x15x26 array with a word in a
        different position each iteration.
    """
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
    """Generator of word masks in all directions."""
    return chain.from_iterable(
        [gen_mask_direction(word, mut) for mut in _directions])


class Masks(object):
    """Represent all possible positions for a given word."""

    def __init__(self, word):
        """See class docstring."""
        self._masks, self._xs, self._ys, self._directions = zip(
            *gen_mask(word))
        self._word = word

    def get_fits(self, wordsearch):
        """Score each position based on a pre-classified wordsearch.

        This is computationally intensive due to having to dot the 15x15
        probability distribution vectors together for each possible position of
        the word on the grid.

        :param wordsearch: The wordsearch object -- MUST be pre-clsasified.

        :return: A MaskFits object encapsulating the score of each mask
        """
        # This will be an nx15x15x26 array where n is the length of self._masks
        matches = np.array(self._masks) * wordsearch.get_classified_array()
        sums = np.sum(matches, axis=(1, 2, 3))
        # If this is too slow, look at np.argpartition
        best = np.argsort(sums)
        getLogger('assignment.wordsearch.masks')\
            .info("Computed fit scores for word = {}"
                  .format(self._word))
        return MaskFits(sums, self._masks, self._xs, self._ys,
                        self._directions, best, self._word)

    def __getitem__(self, index):
        """Get the nth mask, coords and direction.

        Useful for debugging.
        """
        return {'mask': self._masks[index],
                'coords': (self._xs[index], self._ys[index]),
                'direction': self._directions[index]}


class MaskFits(object):
    """Encapsulate the masks for a particular word ranked by score.

    :param sums: The scores for each position and direction combination
    :param masks: The masks for each position and direction combination
    :param xs: The x-coordinates of each position and direction combination
    :param ys: The y-coordinates of each position and direction combination
    :param directions: The directions of each position and direction
        combination
    :param best: The indices of the combinations, sorted by worst to best score
    :param word: The word we're looking for
    """

    def __init__(self, sums, masks, xs, ys, directions, best, word):
        """See class docstring."""
        self._sums = sums
        self._masks = masks
        self._xs = xs
        self._ys = ys
        self._directions = directions
        self._best = best
        self._word = word

    def __getitem__(self, index):
        """Get the nth best-scoring mask, coordinates, direction and score."""
        orderedIndex = self._best[-(index + 1)]
        return {'mask': self._masks[orderedIndex],
                'coords': (self._xs[orderedIndex], self._ys[orderedIndex]),
                'direction': self._directions[orderedIndex],
                'score': self._sums[orderedIndex] / len(self._word),
                'word': self._word}
