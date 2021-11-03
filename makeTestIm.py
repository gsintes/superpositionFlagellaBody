"""Make simple test images for displacement detection"""

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

def make_square_im(
    top_left_corner: Tuple[int, int],
    im_shape: Tuple[int, int],
    square_size: int) -> np.ndarray:
    """Make a white full square in an image."""

    im = np.zeros(im_shape)
    for i in range(im_shape[0]):
        for j in range(im_shape[1]):
            if top_left_corner[0] <= i < top_left_corner[0] + square_size:
                if top_left_corner[1] <= j < top_left_corner[1] + square_size:
                    im[i, j] = 1
    return im

def make_ellipse_im(center: Tuple[int, int],
    im_shape: Tuple[int, int],
    angle: float,
    a: int,
    b: int)-> np.ndarray:
    """Make a white ellispe in an image."""
    x0 = center[0]
    y0 = center[1]
    x0p = np.cos(angle) * x0 + np.sin(angle) * y0
    y0p = - np.sin(angle) * x0 + np.cos(angle) * x0
    im = np.zeros(im_shape)
    for x in range(im_shape[0]):
        for y in range(im_shape[1]):
            xp = np.cos(angle) * x + np.sin(angle) * y
            yp = - np.sin(angle) * x + np.cos(angle) * y
            r = ((xp - x0p) / a) ** 2 + ((yp - y0p) / b) ** 2
            if r <= 1:
                im[x, y] = 1
    return im


if __name__ == "__main__":
    im_shape = (100, 100)
    square_size = 10
    image_init = make_ellipse_im((50, 50), im_shape, np.pi / 2, 40, 30)

    plt.figure()
    plt.imshow(image_init, cmap="gray")
    
    plt.show(block=True)