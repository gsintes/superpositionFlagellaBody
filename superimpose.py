"""Super impose images of the body and the flagella."""

import shutil
from typing import Tuple
import os

import matplotlib.image as mpim
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.filters.thresholding import threshold_otsu

import constants
from mire_info import MireInfo

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

def select_center_image(image: np.ndarray, center: Tuple[int, int], size: int = 100) -> np.ndarray:
    """Return the center part of the image."""
    x_mean = center[0]
    y_mean = center[1]
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

def shift_image(
    image: np.ndarray,
    displacement: Tuple[int, int]) -> np.ndarray:
    """Shift the image and fill boundaries with black."""
    if len(image.shape) == 3:
        shifted = image.copy()
        for i in range(3):
            shifted[:, :, i] = shift_image(image[:, :, i], displacement)
        return shifted
    elif len(image.shape) == 2:
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
    else:
        raise IndexError("Not the good dimension.")


def super_impose_two_im(
    green_im: np.ndarray,
    red_im: np.ndarray,
    displacement: Tuple[int,int]) -> np.ndarray:
    """Super impose the green and red part."""
    shift_red = shift_image(red_im, displacement)
    super_imposed = np.array([shift_red.transpose(),
     green_im.transpose(),
     np.zeros(green_im.shape).transpose()])
    return super_imposed.transpose()


def superposition(image: np.ndarray, mire_info: MireInfo) -> np.ndarray:
    """Superimpose the two colors according to the info of the mire."""
    red_im, green_im = split_image(image, mire_info.middle_line)
    return super_impose_two_im(green_im, red_im, mire_info.displacement)

def folder_superposition(
    folder_im: str,
    folder_save: str,
    mire_info: MireInfo):
    """Run the superposition and save all images in a folder."""

    date_video = folder_im.split("/")
    date_video = date_video[len(date_video) - 1]
    save_dir = os.path.join(folder_save, "Videos_tracking", date_video)
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    image_list = [os.path.join(folder_im, f) for f in os.listdir(folder_im) if (f.endswith(".tif") and not f.startswith("."))]
    for i, im_path in enumerate(image_list):
        im_test = mpim.imread(im_path) 
        im_test = im_test / 2 ** 16
        super_imposed = superposition(im_test, mire_info)
        super_imposed = select_center_image(super_imposed)
        super_imposed = contrast_enhancement(super_imposed)
        plt.figure()
        plt.imshow(super_imposed, cmap="gray")
        mpim.imsave(os.path.join(save_dir, f"{i}.png"), super_imposed)
        plt.close()


if __name__ == "__main__":
    mire_info = MireInfo(constants.MIRE_INFO_PATH)
    # parent_folder = "/Volumes/GUILLAUME/Ficoll Marty/Ficoll17%_20-11-05_1uLbactos_TRACKING"
    # list_dir = [f for f in os.listdir(parent_folder) if not f.startswith(".")]
    # for folder in list_dir[0: 1]:
        # folder_superposition(os.path.join(parent_folder, folder), "/Users/sintes/Desktop", mire_info)
    folder_superposition(constants.FOLDER, "/Users/sintes/Desktop", mire_info)