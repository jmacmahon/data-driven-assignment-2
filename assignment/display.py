import matplotlib.pyplot as plt
from matplotlib.cm import gray
import numpy as np

def show_image(pixels, shape):
    shaped_matrix = np.reshape(pixels, shape, order='F')
    return plt.matshow(shaped_matrix, cmap=gray)

def show_letter(pixels):
    return show_image(pixels, (30, 30))

def show_wordsearch(pixels):
    return show_image(pixels, (450, 450))
