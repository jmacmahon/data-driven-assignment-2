import matplotlib.pyplot as plt
from matplotlib.cm import gray
from math import sqrt, floor
import numpy as np


def show_image(pixels, shape):
    shaped_matrix = np.reshape(pixels, shape, order='F')
    return plt.imshow(shaped_matrix, cmap=gray)


def pixel_line(start, end, rad=2):
    base = np.zeros((450, 450, 4))
    sx, sy = start
    ex, ey = end
    steps = int(max(abs(ex - sx), abs(ey - sy)))
    xstep = (ex - sx)/steps
    ystep = (ey - sy)/steps
    currentx, currenty = start
    yellow_pixel = np.array([1, 1, 0, 1])
    for _ in range(steps):
        icurrentx, icurrenty = int(currentx), int(currenty)
        base[max(0, icurrenty - rad):min(450, icurrenty + rad),
             max(0, icurrentx - rad):min(450, icurrentx + rad)] = yellow_pixel
        currentx += xstep
        currenty += ystep
    return base


def letter_line(x, y, direction, length, rad=2):
    start = (x*30, y*30)
    end = (((length - 1) * direction[0] + x) * 30,
           ((length - 1) * direction[1] + y) * 30)
    return pixel_line(start, end, rad=rad)


def show_letter(pixels):
    return show_image(pixels, (30, 30))


def show_wordsearch(pixels):
    return show_image(pixels, (450, 450))
