"""Super impose images of the body and the flagella."""

from typing import Tuple, List
from statistics import median

import matplotlib.image as mpim
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters.thresholding import threshold_otsu
from skimage import exposure
from scipy.signal import correlate2d

# MIRE_PATH = "/Volumes/GUILLAUME/2021-10-08_chaintracking/2021-10-08_15h56m14sMyre/Image0007645.tif"
MIRE_PATH = "/Volumes/GUILLAUME/Ficoll Marty/2020-11-05_13h43m12s_mire/Image0574023.tif"

IM_SIZE = (1024, 1024)


def moving_average(array: np.ndarray, averaging_length: int) -> np.ndarray:
    """Does a moving average of array with a given averaging length."""
    return np.convolve(array, np.ones(averaging_length), "valid") / averaging_length


def find_separation(mire_im: np.ndarray, visualization: bool=False) -> int:
    """Open the mire image and find the separation line."""
    loc_profiles = range(10, IM_SIZE[0] - 10, 10)
    separators = []
    for loc_profile in loc_profiles:
        profile = mire_im[:, loc_profile] / max(mire_im[:, loc_profile])
        smooth_prof = moving_average(profile, 20)
        diff = smooth_prof[401: 651] - smooth_prof[400 : 650]
        separators.append(405 + list(diff).index(min(diff)))
    separation = median(separators)
    if visualization:
        plt.figure()
        plt.plot(profile)
        plt.plot(smooth_prof)
        plt.plot([separation, separation], [min(profile), max(profile)], "-r")

        plt.figure()
        plt.imshow(mire_im, cmap="gray")
        plt.plot([0, IM_SIZE[0]], [separation, separation], "-r")
        plt.xlim([0, IM_SIZE[0]])
    plt.show()
    return int(separation)


def contrast_enhancement(image: np.ndarray) -> np.ndarray:
    """Enhance the contrast of the image."""
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    return img_rescale


def binarize(im: np.ndarray) -> np.ndarray:
    """Binarize an image using Otsu's method."""
    threshold = threshold_otsu(im)
    bin_im = (im > threshold) * 1
    return bin_im


def split_image(
    image: np.ndarray,
    separation: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split the image at the separation and return the two images off the same size, complete by zeros."""
    red_im: np.ndarray = contrast_enhancement(image[:separation, :])
    green_im: np.ndarray = contrast_enhancement(image[separation:, :])
    diff_sep = 2 * (separation - image.shape[0] // 2)
    to_add = np.zeros((diff_sep, image.shape[1]))

    if diff_sep > 0:
        green_im = np.concatenate((green_im, to_add))
    if diff_sep < 0:
        red_im = np.concatenate((to_add, red_im))
    return red_im, green_im


def select_center_image(image: np.ndarray) -> np.ndarray:
    """Return the center part of the image."""
    x_mean = image.shape[0] // 2
    y_mean = image.shape[1] // 2
    # return image[x_mean - 200 : x_mean + 200, y_mean - 200 : y_mean + 200]
    return image


def find_displacement(
    green_mire: np.ndarray,
    red_mire: np.ndarray,
    visualization: bool = False) -> Tuple[int, int]:
    """Find the displacement of the two pictures."""

    cross_corr = correlate2d(
        binarize(select_center_image(red_mire)),
        binarize(select_center_image(green_mire)))
    
    i, j = np.unravel_index(cross_corr.argmax(), cross_corr.shape)
    delta_x = 1 + i - select_center_image(red_mire).shape[0]
    delta_y =  1 + j - select_center_image(red_mire).shape[1]

    if visualization:
        plt.figure()
        plt.imshow(cross_corr, cmap="gray")
        plt.plot(i, j, "*r")
    return (delta_x, delta_y)

def shift_image(
    image: np.ndarray,
    displacement: Tuple[int, int]) -> np.ndarray:
    """Shift the image and fill boundaries with black."""
    delta_x = displacement[0]
    delta_y = displacement[1]
    if delta_x > 0:
        image = image[delta_x:, :]
        image = np.concatenate((image, np.zeros((delta_x, image.shape[1]))), axis=0)
    if delta_x < 0:
        image = image[:delta_x, :]
        image = np.concatenate((np.zeros((-delta_x, image.shape[1])), image), axis=0)

    if delta_y < 0: 
        image = image[:, :delta_y]
        image = np.concatenate((np.zeros((image.shape[0], -delta_y, )), image), axis=1)
    if delta_y > 0:
        image = image[:, delta_y:]
        image = np.concatenate((image, np.zeros((image.shape[0], delta_y))), axis=1)
    return image


def super_impose_two_im(
    green_im: np.ndarray,
    red_im: np.ndarray,
    displacement: Tuple[int,int]) -> np.ndarray:
    """Super impose the green and red part."""
    shift_red = shift_image(red_im, displacement)
    # shift_red = red_im
    super_imposed = np.array([shift_red.transpose(),
     green_im.transpose(),
     np.zeros(green_im.shape).transpose()])
    return super_imposed.transpose()

def manual_find_displacement(
    green_mire: np.ndarray,
    red_mire) -> Tuple[int, int]:
    """Manually find the displacement in the image by clicking on a point."""
    plt.figure()
    plt.imshow(green_mire)
    point_one = plt.ginput(1)[0]
    plt.close()
    plt.figure()
    plt.imshow(red_mire)
    point_two = plt.ginput(1)[0]
    plt.close()
    print(point_one)
    print(point_two)
    delta_x = int(point_two[0] - point_one[0])
    delta_y = int(point_two[1] - point_one[1])
    return (delta_y, delta_x)




if __name__ == "__main__":
    mire_im = mpim.imread(MIRE_PATH) 
    mire_im = mire_im / np.amax(mire_im)
    middle_line = find_separation(mire_im, visualization=False)
    red_mire, green_mire = split_image(mire_im, middle_line)

    displacement = manual_find_displacement(green_mire, red_mire)
    super_imposed = super_impose_two_im(green_mire, red_mire, displacement)

    plt.figure()
    plt.imshow(super_imposed)
    plt.show(block=True)
    

