from pickle import load
from .data import Data

DEFAULT_PATH = "assignment2.pkl"


def load_data(pickle_path):
    with open(pickle_path, "rb") as f:
        data = load(f)
    return Data(data)
