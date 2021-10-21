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

if __name__ == "__main__":
    im_shape = (100, 100)
    square_size = 10
    image_init = make_square_im((0, 0), im_shape, square_size)
    image_sec = make_square_im((10, 40), im_shape, square_size)

    plt.figure()
    plt.imshow(image_init, cmap="gray")
    plt.figure()
    plt.imshow(image_sec, cmap="gray")
    plt.show(block=True)