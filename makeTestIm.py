"""Make simple test images for displacement detection"""

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

class Mask():
    def __init__(self, center: Tuple[int, int], angle: float) -> None:
        self.center = center
        self.angle = angle
    
    def make_im(self, im_shape: Tuple[int, int]) -> np.ndarray:
        raise NotImplementedError

class Ellipse(Mask):
    """Make an ellipsoidal mask."""
    def __init__(self,
    center: Tuple[int, int],
    angle: float,
    a: int,
    b: int) -> None:
        super().__init__(center, angle)
        self.a = a
        self.b = b

    def make_im(self,im_shape: Tuple[int, int]) -> np.ndarray:
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

class Rectangle(Mask):
    """Make a rectangular mask."""
    def __init__(self,
        length: int,
        width: int,
        center: Tuple[int, int],
        angle: float) -> None:
        self.length = length
        self.width = width
        super().__init__(center, angle)
    
    def make_im(self,im_shape: Tuple[int, int])-> np.ndarray:
        """Make a white ellispe in an image."""
        x0 = self.center[0]
        y0 = self.center[1]

        x0p = np.cos(self.angle) * x0 + np.sin(self.angle) * y0
        y0p = - np.sin(self.angle) * x0 + np.cos(self.angle) * y0

        im = np.zeros(im_shape)
        for x in range(im_shape[0]):
            for y in range(im_shape[1]):
                xp = np.cos(self.angle) * x + np.sin(self.angle) * y
                yp = - np.sin(self.angle) * x + np.cos(self.angle) * y 
                if - self.length // 2 < (xp - x0p) < self.length // 2:
                    if - self.width // 2 < yp - y0p < self.width // 2:
                        im[x, y] = 1
        return im
    
    def border(self,im_shape: Tuple[int, int])-> Tuple[List[int], List[int]]:
        """Give the border of the rectangle."""
        x0 = self.center[0]
        y0 = self.center[1]

        x0p = np.cos(self.angle) * x0 + np.sin(self.angle) * y0
        y0p = - np.sin(self.angle) * x0 + np.cos(self.angle) * y0
        X = []
        Y = []
        for x in range(im_shape[0]):
            for y in range(im_shape[1]):
                xp = np.cos(self.angle) * x + np.sin(self.angle) * y
                yp = - np.sin(self.angle) * x + np.cos(self.angle) * y 
                if self.length / 2 - 0.5 <= np.abs(xp - x0p) <= self.length / 2 + 0.5:
                    if np.abs(yp - y0p) <= self.width // 2:
                        X.append(x)
                        Y.append(y)
                if self.width / 2 - 0.5 <= np.abs(yp - y0p) <= self.width / 2 + 0.5:
                    if np.abs(xp - x0p) <= self.length // 2 + 0.5:
                        X.append(x)
                        Y.append(y)
        return X, Y


if __name__ == "__main__":
    im_shape = (100, 100)
    square_size = 10
    ellipse = Rectangle(center=(50, 50), angle=np.pi / 2, length=40, width=10)
    image_init = ellipse.make_im(im_shape)

    plt.figure()
    plt.imshow(image_init, cmap="gray")
    plt.axis("equal")
    plt.show(block=True)