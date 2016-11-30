import matplotlib.pyplot as plt
from matplotlib.cm import gray
import numpy as np

def show_letter(pixels):
    shaped_matrix = np.reshape(pixels, (30, 30), order='F')
    return plt.matshow(shaped_matrix, cmap=gray)
