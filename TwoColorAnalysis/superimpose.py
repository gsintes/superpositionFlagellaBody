"""Super impose images of the body and the flagella."""

import shutil
from typing import Tuple
import os

import matplotlib.image as mpim
import numpy as np
from skimage import exposure
from skimage.filters.thresholding import threshold_otsu

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
    """Split the image at the separation and return the two images at the size of the smaller."""
    diff_sep = 2 * (separation - image.shape[0] // 2)
    if diff_sep > 0:
        top_im = image[diff_sep:separation, :]
        bottom_im = image[separation:, :]
    if diff_sep < 0:
        top_im = image[:separation, :]
        bottom_im = image[separation:-diff_sep, :]
    return top_im, bottom_im

def shift_image(
    bottom_image: np.ndarray,
    top_image: np.ndarray,
    displacement: Tuple[int, int]) -> np.ndarray:
    """Shift the image and fill boundaries with black."""
    assert len(bottom_image.shape) == 2
    assert len(top_image.shape) == 2
    delta_x = displacement[0]
    delta_y = displacement[1]
    if delta_x > 0:
        bottom_image = bottom_image[delta_x:, :]
        top_image = top_image[:-delta_x, :]
    if delta_x < 0:
        bottom_image = bottom_image[:delta_x, :]
        top_image = top_image[-delta_x:, :]

    if delta_y < 0:
        bottom_image = bottom_image[:, :delta_y]
        top_image = top_image[:, -delta_y:]
    if delta_y > 0:
        bottom_image = bottom_image[:, delta_y:]
        top_image = top_image[:, delta_y:]
    return bottom_image, top_image



def super_impose_two_im(
    bottom_im: np.ndarray,
    top_im: np.ndarray,
    displacement: Tuple[int,int]) -> np.ndarray:
    """Super impose the green and red part."""
    bottom_im, top_im = shift_image(bottom_im, top_im, displacement)
    super_imposed = np.array([top_im.transpose(),
     bottom_im.transpose(),
     np.zeros(bottom_im.shape).transpose()])
    return super_imposed.transpose()


def superposition(image: np.ndarray, mire_info: MireInfo) -> np.ndarray:
    """Superimpose the two colors according to the info of the mire."""
    top_im, bottom_im = split_image(image, mire_info.middle_line)
    return super_impose_two_im(top_im, bottom_im, mire_info.displacement)


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
    for i, im_path in enumerate(image_list[:-1]):
        im_test = mpim.imread(im_path)
        im_test = im_test / 2 ** 16
        super_imposed = superposition(im_test, mire_info)
        super_imposed = contrast_enhancement(super_imposed)
        mpim.imsave(os.path.join(save_dir, f"{i}.png"), super_imposed)


if __name__ == "__main__":
    mire_info = MireInfo("/Volumes/Chains/2colors0502/2024-05-02_17h43m35s_calib/mire_info.json")
    parent_folder = "/Volumes/Chains/2colors0502"
    list_dir = [f for f in os.listdir(parent_folder) if not("calib" in f) and f.startswith("202") and os.path.isdir(os.path.join(parent_folder, f))]
    for folder in list_dir:
        folder_superposition(os.path.join(parent_folder, folder), parent_folder, mire_info)
    # folder_superposition(constants.FOLDER, "/Volumes/Chains/Tracking_23-12-08/superimposed", mire_info)