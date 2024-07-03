"""Detect the body using convolution techniques."""

from typing import Tuple, List
import os

import numpy as np
import matplotlib.image as mpim
from matplotlib import pyplot as plt

from makeTestIm import Mask, Rectangle

import superimpose
import mire_info
import utils
import angleBodyFlagella as abf

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
    if wtot != 0:
        return (int(x_sum / wtot), int(y_sum / wtot))
    return (100, 100)


class Convoluter:
    """Does convolution between a mask and an image."""
    def __init__(self, image: np.ndarray, mask: Mask) -> None:
        self.image = image
        self.mask = mask
        self.mask.center = (image.shape[0] // 2, image.shape[1] // 2)
        self.mask_im = self.mask.make_im(self.image.shape)

    def __call__(self, visualization : bool = False) -> np.ndarray:
        """Does the convolution product between the image and the mask."""
        ft_im = np.fft.rfft2(self.image)
        ft_mask = np.fft.rfft2(self.mask_im)
        ft_conv = ft_mask * ft_im
        conv = np.fft.irfft2(ft_conv, s=self.image.shape)
        conv_shift = np.zeros(conv.shape)
        for x in range(conv.shape[0]):
            for y in range(conv.shape[1]):
                conv_shift[(x + self.mask.center[0]) % conv.shape[0], (y + self.mask.center[1]) % conv.shape[1]] = conv[x, y]
        if visualization:
            plt.figure()
            plt.title(np.floor(180 * self.mask.angle / np.pi))
            plt.imshow(conv_shift.transpose())
        return conv_shift

class BodyDetection:
    """Perform body detection"""
    def __init__(self, image: np.ndarray, a: int, b: int) -> None:
        self.angle = 0 # in degree
        self.image = image
        self.a = a
        self.b = b

    def detection(self, step: int = 10, lim: int = 180, visualization: bool = False) -> None:
        """Does a detection of the object with step in angle."""
        angles: List[int] = []
        convolutions: List[np.array] = []
        while self.angle < lim:
            convoluter = Convoluter(self.image, Rectangle(self.a, self.b, (0, 0), self.angle * np.pi / 180))
            angles.append(self.angle)
            convolutions.append(convoluter(visualization=visualization))
            self.angle += step
        sum = np.zeros(self.image.shape)
        for conv in convolutions:
            sum += conv
        bin_sum = utils.li_binarization(sum)
        self.center = center_of_mass(bin_sum)
        self.best_angle = 0
        max = 0
        for i, conv in enumerate(convolutions):
            val = np.mean(conv[self.center[0] - 1 : self.center[0] + 1, self.center[1] - 1 : self.center[1] + 1])
            if val > max:
                max = val
                self.best_angle = angles[i]

        if visualization:
            plt.figure()
            plt.title("Sum of convolutions")
            plt.imshow(sum.transpose())
            plt.plot(self.center[0], self.center[1], "ko", markersize=3)

    def __call__(self, visualization=False) -> Mask:
        """Does a detection of the body with rough then precise angle detection."""
        self.detection()
        self.angle = self.best_angle - 5
        self.detection(step=1, lim=self.best_angle + 5, visualization=False)
        if visualization:
            rectangle = Rectangle(self.a, self.b, self.center, np.pi * self.best_angle / 180)
            X, Y = rectangle.border(self.image.shape)
            plt.figure()
            plt.imshow(self.image.transpose(), cmap="gray")
            plt.plot(X, Y, ".r", markersize=2)
            plt.show(block=True)

        return Rectangle(self.a, self.b, self.center, np.pi * self.best_angle / 180)

if __name__ == "__main__":
    folder_up = "/Users/sintes/Desktop/NASGuillaume/SwimmingPVP360/SwimmingPVP_23-07-25"
    mire_info_path = f"{folder_up}/2023-07-25_18h06m22s_calib/mire_info.json"
    folder = os.path.join(folder_up, "2023-07-25_18h35m40s")
    mire = mire_info.MireInfo(mire_info_path)

    image_list = [os.path.join(folder , f) for f in os.listdir(folder) if (f.endswith(".tif") and not f.startswith("."))]
    i = 1757
    im_test = mpim.imread(image_list[i])
    info = abf.Info((folder), 80, mire)
    super_imposed = superimpose.superposition(im_test, info.mire_info)
    center_track = (int(info.track_data["center_x"][i]) - info.mire_info.middle_line, int(info.track_data["center_y"][i]))
    center = abf.get_center_body(super_imposed, center_track)
    # print(center)
    super_imposed = superimpose.select_center_image(
            super_imposed,
            center=center,
            size=100)
    image = super_imposed[:, :, 1]

    bd = BodyDetection(image, 40, 7)
    bd(visualization=True)

    # res = []
    # for angle in range(0, 180):
    #     print(angle)
    #     IMAGE = Rectangle(length=40, width=7, center=(50, 70), angle=angle * np.pi / 180).make_im((200, 200))

    #     bd = BodyDetection(IMAGE, 40, 6)
    #     res.append(int(bd(visualization=False).angle * 180 / np.pi))
    # delta = np.array([res[i] - i for i in range(0, 180)])
    # print(res)
    # print(delta)
    # print(np.mean(delta), np.sqrt(np.mean(delta ** 2)))
