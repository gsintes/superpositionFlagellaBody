"""Tools for superposition."""

from typing import Tuple

import numpy as np
from skimage import exposure
from skimage.filters.thresholding import threshold_otsu


def contrast_enhancement(image: np.ndarray) -> np.ndarray:
    """Enhance the contrast of the image."""
    if len(image.shape) == 3:
        image_enhanced = image.copy()
        for i in range(3):
            image_enhanced[:, :, i] = contrast_enhancement(image_enhanced[:, :, i])
        return image_enhanced
    elif len(image.shape) == 2:
        p2, p98 = np.percentile(image, (1, 99))
        img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))        
        return img_rescale
    else:
        raise IndexError("Not the good dimension.")


def moving_average(array: np.ndarray, averaging_length: int) -> np.ndarray:
    """Does a moving average of array with a given averaging length."""
    return np.convolve(array, np.ones(averaging_length), "valid") / averaging_length


def binarize(im: np.ndarray) -> np.ndarray:
    """Binarize an image using Otsu's method."""
    threshold = threshold_otsu(im)
    bin_im = (im > threshold) * 1
    return bin_im

def select_center_image(image: np.ndarray, size: int = 100) -> np.ndarray:
    """Return the center part of the image."""
    x_mean = image.shape[0] // 2
    y_mean = image.shape[1] // 2
    if len(image.shape) == 2:
        return image[x_mean - size : x_mean + size, y_mean - size : y_mean + size]
    return image[x_mean - size : x_mean + size, y_mean - size : y_mean + size, :]

def split_image(
    image: np.ndarray,
    separation: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split the image at the separation and return the two images off the same size, complete by zeros."""
    red_im = image[:separation, :]
    green_im = image[separation:, :]
    diff_sep = 2 * (separation - image.shape[0] // 2)
    to_add = np.zeros((np.abs(diff_sep), image.shape[1]))

    if diff_sep > 0:
        green_im = np.concatenate((green_im, to_add))
    if diff_sep < 0:
        red_im = np.concatenate((to_add, red_im))
    return red_im, green_im