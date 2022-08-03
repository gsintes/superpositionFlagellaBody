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

class Ellipse:
    def __init__(self,
    center: Tuple[int, int],
    angle: float,
    a: int,
    b: int) -> None:
        self.center = center
        self.angle = angle
        self.a = a
        self.b = b

    def make_im(self,im_shape: Tuple[int, int])-> np.ndarray:
        """Make a white ellispe in an image."""
        x0 = self.center[0]
        y0 = self.center[1]
        x0p = np.cos(self.angle) * x0 + np.sin(self.angle) * y0
        y0p = - np.sin(self.angle) * x0 + np.cos(self.angle) * x0
        im = np.zeros(im_shape)
        for x in range(im_shape[0]):
            for y in range(im_shape[1]):
                xp = np.cos(self.angle) * x + np.sin(self.angle) * y
                yp = - np.sin(self.angle) * x + np.cos(self.angle) * y
                r = ((xp - x0p) / self.a) ** 2 + ((yp - y0p) / self.b) ** 2
                if r <= 1:
                    im[x, y] = 1
        return im

class Rectangle:
    def __init__(self,
        length: int,
        width: int,
        center: Tuple[int, int],
        angle: float) -> None:
        self.length = length
        self.width = width
        self.center = center
        self.angle = angle
    
    def make_im(self,im_shape: Tuple[int, int])-> np.ndarray:
        """Make a white ellispe in an image."""
        x0 = self.center[0]
        y0 = self.center[1]

        x0p = np.cos(self.angle) * x0 + np.sin(self.angle) * y0
        y0p = - np.sin(self.angle) * x0 + np.cos(self.angle) * x0

        im = np.zeros(im_shape)
        for x in range(im_shape[0]):
            for y in range(im_shape[1]):
                xp = np.cos(self.angle) * x + np.sin(self.angle) * y
                yp = - np.sin(self.angle) * x + np.cos(self.angle) * y 
                if -self.length < xp - x0p < self.length // 2:
                    if -self.width < yp - y0p < self.width // 2:
                        im[x, y] = 1
        return im



if __name__ == "__main__":
    im_shape = (100, 100)
    square_size = 10
    ellipse = Rectangle(center=(50, 50), angle=np.pi / 2, length=40, width=10)
    image_init = ellipse.make_im(im_shape)

    plt.figure()
    plt.imshow(image_init, cmap="gray")
    plt.axis("equal")
    plt.show(block=True)