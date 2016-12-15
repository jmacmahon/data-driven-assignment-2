"""A collection of helper display functions using a pyplot backend."""

import matplotlib.pyplot as plt
from matplotlib.cm import gray
from math import sqrt, floor
import numpy as np


def show_image(pixels, shape):
    """Display the provided pixels in the provided shape."""
    shaped_matrix = np.reshape(pixels, shape, order='F')
    return plt.imshow(shaped_matrix, cmap=gray)


def show_letter(pixels):
    """Display the provided pixels in the standard letter shape."""
    return show_image(pixels, (30, 30))


def show_wordsearch(pixels):
    """Display the provided pixels in the standard wordsearch shape."""
    return show_image(pixels, (450, 450))


def pixel_line(start, end, rad=2):
    """Create a line from at the specified pixel coordinates with a radius.

    :param start: The starting position as pixel coordinates
    :param end: The ending position as pixel coordinates
    :param rad: The radius of the line in pixels

    :return: A 450x450x4 RGBA image array with a yellow line drawn on a
        transparent background
    """
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
    """Create a line at the specified letter-grid coordinates.

    :param x: The starting x-coordinate (0 <= x < 15)
    :param y: The starting y-coordinate (0 <= y < 15)
    :param direction: A tuple giving the direction of the line
    :param length: The length of the line in letters
    :param rad: The radius of the line in pixels

    :return: A 450x450x4 RGBA image array with a yellow line drawn on a
        transparent background
    """
    start = (x*30, y*30)
    end = (((length - 1) * direction[0] + x) * 30,
           ((length - 1) * direction[1] + y) * 30)
    return pixel_line(start, end, rad=rad)


def draw_line_from_word(word, coords, direction, rad=2):
    """Create a line from a single word-solution to a wordsearch.

    :param word: The word as a string
    :param coords: The (x, y) coordinate of the first letter of the word
    :param direction: A tuple giving the direction of the solution
    :param rad: Thre radius of the line in pixels

    :return: A 450x450x4 RGBA image array with a yellow line drawn on a
        transparent background
    """
    x, y = coords
    x += 0.5
    y += 0.5
    return letter_line(x, y, direction, len(word), rad=rad)


def draw_lines(words, rad=2):
    """Draw a line for each word-solution to a wordsearch and show the result.

    :param words: The word-solutions as a `dict`:
        `{'word': (start, end, direction)}`
    :param rad: The radius of the line in pixels
    """
    lines = [draw_line_from_word(word, start, direction, rad=rad) for
             (word, (start, end, direction)) in words.items()]
    [plt.imshow(line, alpha=0.5) for line in lines]
