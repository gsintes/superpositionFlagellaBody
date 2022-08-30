"""Detect the body using convolution techniques."""

from curses.textpad import rectangle
import os
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpim

from utils import timeit
from makeTestIm import Mask, Rectangle
import makeTestIm as masks
import angleBodyFlagella as abf
import constants
import superimpose

def center_of_mass(image: np.ndarray) -> Tuple[int, int]:
    """Return the center of mass of an image weighted by pixel intensity."""
    x_sum = 0
    y_sum = 0
    wtot = 0
    for x in range(image.shape[0]):
        for y in range(image.shape[0]):
            x_sum += x * image[x, y]
            y_sum +=  y * image[x, y]
            wtot += image[x, y]
    return (int(x_sum / wtot), int(y_sum / wtot))


class Convoluter:
    """Does convolution between a mask and an image."""
    def __init__(self, image: np.ndarray, mask: Mask) -> None:
        self.image = image
        self.mask = mask
        self.mask.center = (image.shape[0] // 2, image.shape[1] // 2)
        self.mask_im = self.mask.make_im(self.image.shape)
        
    def __call__(self) -> np.ndarray:
        """Does the convolution product between the image and the mask."""
        ft_im = np.fft.rfft2(self.image)
        ft_mask = np.fft.rfft2(self.mask_im)
        ft_conv = ft_mask * ft_im
        conv = np.fft.irfft2(ft_conv, s=self.image.shape) 
        conv_shift = np.zeros(conv.shape)
        for x in range(conv.shape[0]):
            for y in range(conv.shape[1]):
                conv_shift[(x + self.mask.center[0]) % conv.shape[0], (y + self.mask.center[1]) % conv.shape[1]] = conv[x, y]
        return conv_shift

class BodyDetection:
    """Perform body detection"""
    def __init__(self, image: np.ndarray, a: int, b: int) -> None:
        self.angle = 0 # in degree
        self.image = image
        self.a = a 
        self.b = b

    def detection(self, step=10, lim=180) -> None:
        """Does a detection of the object with step in angle."""
        angles = []
        convolutions = []
        while self.angle < lim:
            convoluter = Convoluter(self.image, Rectangle(self.a, self.b, (0, 0), self.angle * np.pi / 180))
            angles.append(self.angle)
            convolutions.append(convoluter())
            self.angle += step 
        sum = np.zeros(self.image.shape)
        for conv in convolutions:
            sum += conv
        self.center = center_of_mass(sum)
        self.best_angle = 0
        max = 0
        for i, conv in enumerate(convolutions):
            val = np.mean(conv[self.center[0] - 1 : self.center[0] + 1, self.center[1] - 1 : self.center[1] + 1])
            if val > max:
                max = val
                self.best_angle = angles[i]
    @timeit
    def __call__(self, visualization=False) -> int:
        """Does a detection of the body with rough then precise angle detection."""
        self.detection()
        self.angle = self.best_angle - 10
        self.detection(step=1, lim=self.best_angle + 10)

        if visualization:
            rectangle = Rectangle(self.a, self.b, self.center, np.pi * self.best_angle / 180)
            X, Y = rectangle.border(self.image.shape)
            plt.figure()
            plt.imshow(self.image.transpose(), cmap="gray")
            plt.plot(X, Y, ".r", markersize=2)
            plt.savefig(f"/Users/sintes/Desktop/Test/{angle}.png")
            plt.close()
            # plt.show(block=True)

        return self.best_angle

if __name__ == "__main__":
    mire_info = superimpose.MireInfo(constants.MIRE_INFO_PATH)

    image_list = [os.path.join(constants.FOLDER, f) for f in os.listdir(constants.FOLDER) if (f.endswith(".tif") and not f.startswith("."))]
    image = mpim.imread(image_list[2])
    super_imposed = superimpose.shift_image(superimpose.superposition(image, mire_info),(0, 0))
    super_imposed = superimpose.select_center_image(super_imposed, 100) 

    image = super_imposed[:, :, 1]
    res = []
    for angle in range(0, 180):
        print(angle)
        IMAGE = masks.Rectangle(length=40, width=6, center=(100, 100), angle=angle * np.pi / 180).make_im((200, 200))

        bd = BodyDetection(IMAGE, 40, 6)
        res.append(bd(visualization=True))
    delta = np.array([res[i] - i for i in range(0, 180)])
    print(res)
    print(delta)
    print(np.mean(delta), np.sqrt(np.mean(delta ** 2)))