"""Some helper functions for loading the example assignment data."""

from pickle import load
from .data import Data

DEFAULT_PATH = "assignment2.pkl"

CORRECT_COORDS = {
    'barry': ((12, 12), (11, 6)),
    'beardshaw': ((0, 12), (8, 2)),
    'bridgeman': ((4, 10), (12, 0)),
    'brown': ((2, 6), (1, 0)),
    'cane': ((0, 2), (-1, 5)),
    'crowe': ((10, 3), (4, 2)),
    'don': ((6, 12), (2, 11)),
    'fish': ((3, 4), (6, 7)),
    'flowerdew': ((14, 2), (4, 10)),
    'hoare': ((11, 8), (10, 12)),
    'jekyll': ((0, 11), (-1, 4)),
    'jellicoe': ((14, 4), (5, 11)),
    'kent': ((7, 14), (2, 13)),
    'langley': ((8, 0), (0, -1)),
    'nesfield': ((9, 14), (0, 5)),
    'paine': ((14, 1), (8, 5)),
    'paxton': ((14, 5), (13, 10)),
    'peto': ((1, 10), (0, 5)),
    'repton': ((1, 6), (0, -1)),
    'robinson': ((1, 0), (8, 7)),
    'roper': ((7, 1), (11, 0)),
    'shenstone': ((13, 14), (12, 4)),
    'vanbrugh': ((14, 3), (5, 10)),
    'wright': ((9, 2), (2, 1))
}


def load_data(pickle_path=DEFAULT_PATH):
    """Load the example data from the provided path.

    :param pickle_path: The path to load from

    :return: A `data.Data` object containing the example data
    """
    with open(pickle_path, "rb") as f:
        data = load(f)
    data['correct_coords'] = CORRECT_COORDS
    return Data(data)
