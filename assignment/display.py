import matplotlib.pyplot as plt
from matplotlib.cm import gray
import numpy as np

def show_image(pixels, shape):
    shaped_matrix = np.reshape(pixels, shape, order='F')
    return plt.imshow(shaped_matrix, cmap=gray)

def line(start, end, rad=2):
    base = np.zeros((450, 450, 4))
    sx, sy = start
    ex, ey = end
    steps = int(max(abs(ex - sx), abs(ey - sy)))
    xstep = (ex - sx)/steps
    ystep = (ey - sy)/steps
    currentx, currenty = start
    for _ in range(steps):
        icurrentx, icurrenty = int(currentx), int(currenty)
        for fillx in range(max(0, icurrentx - rad), min(450, icurrentx + rad)):
            for filly in range(max(0, icurrenty - 5), min(450, icurrenty + 5)):
                base[filly, fillx] = np.array([1, 1, 0, 1])
        currentx += xstep
        currenty += ystep
    return base

def show_letter(pixels):
    return show_image(pixels, (30, 30))

def show_wordsearch(pixels):
    return show_image(pixels, (450, 450))
