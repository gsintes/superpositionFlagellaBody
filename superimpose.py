"""Super impose images of the body and the flagella."""

import shutil
from typing import Tuple
from statistics import median
import json
import os

import matplotlib.image as mpim
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.filters.thresholding import threshold_otsu
from scipy.signal import correlate2d

import constants

class MireInfo:
    def __init__(self, *args) -> None:
        if len(args) == 2:
            self.middle_line = args[0]
            self.displacement = args[1]
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, dict):
                self.middle_line = arg["middle_line"]
                self.displacement = arg["displacement"]
            if isinstance(arg, str):
                with open(arg) as f:
                    data = json.load(f)
                    self.middle_line = data["middle_line"]
                    self.displacement = data["displacement"]

    def delta_x(self) -> int:
        """Return the displacement in x"""
        return self.displacement[0]
    
    def delta_y(self) -> int:
        """Return the displacement in y"""
        return self.displacement[1]
    
    def save(self, file: str) -> None:
        """Save the mire info in a json file."""
        with open(file, "w", encoding="utf-8") as outfile:
            outfile.write("")
            json.dump(self.__dict__, outfile, indent=4)

    def __repr__(self) -> str:
        return f"Mire info:\n middle_line: {self.middle_line}\n Displacement: {self.displacement}"


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


def find_separation(mire_im: np.ndarray, visualization: bool=False) -> int:
    """Open the mire image and find the separation line."""
    loc_profiles = range(10, constants.IM_SIZE[0] - 10, 10)
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
        plt.plot([0, constants.IM_SIZE[0]], [separation, separation], "-r")
        plt.xlim([0, constants.IM_SIZE[0]])
    return int(separation)


def binarize(im: np.ndarray) -> np.ndarray:
    """Binarize an image using Otsu's method."""
    threshold = threshold_otsu(im)
    bin_im = (im > threshold) * 1
    return bin_im


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


def select_center_image(image: np.ndarray, size: int = 100) -> np.ndarray:
    """Return the center part of the image."""
    x_mean = image.shape[0] // 2
    y_mean = image.shape[1] // 2
    if len(image.shape) == 2:
        return image[x_mean - size : x_mean + size, y_mean - size : y_mean + size]
    return image[x_mean - size : x_mean + size, y_mean - size : y_mean + size, :]


def find_displacement(
    green_mire: np.ndarray,
    red_mire: np.ndarray,
    visualization: bool = False) -> Tuple[int, int]:
    """Find the displacement of the two pictures."""

    cross_corr = correlate2d(
        binarize(select_center_image(red_mire, 100)),
        binarize(select_center_image(green_mire, 100)))
    
    i, j = np.unravel_index(cross_corr.argmax(), cross_corr.shape)
    delta_x = 1 + i - select_center_image(red_mire, 100).shape[0]
    delta_y =  1 + j - select_center_image(red_mire, 100).shape[1]

    if visualization:
        plt.figure()
        plt.imshow(cross_corr, cmap="gray")
        plt.plot(i, j, "*r")
    return (delta_x, delta_y)


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


def manual_find_displacement(
    green_mire: np.ndarray,
    red_mire) -> Tuple[int, int]:
    """Manually find the displacement in the image by clicking on a point."""
    fig = plt.figure()
    plt.imshow(select_center_image(green_mire, 100))
    point_one = plt.ginput(1)[0]
    plt.close(fig)
    fig = plt.figure()
    plt.imshow(select_center_image(red_mire, 100))
    point_two = plt.ginput(1)[0]
    plt.close(fig)

    delta_x = int(point_two[0] - point_one[0])
    delta_y = int(point_two[1] - point_one[1])
    return (delta_y, delta_x)


def mire_analysis(mire_path: str, visualization: bool = False) -> MireInfo:
    """Perform the mire analysis"""
    mire_im = mpim.imread(mire_path) 
    mire_im = mire_im / 2 ** 16
    middle_line = find_separation(mire_im, visualization)
    red_mire, green_mire = split_image(mire_im, middle_line)

    displacement = manual_find_displacement(green_mire, red_mire)
    super_imposed = super_impose_two_im(green_mire, red_mire, displacement)

    if visualization:
        plt.figure()
        plt.imshow(contrast_enhancement(super_imposed))
        plt.show(block=True)
    res = MireInfo(middle_line, displacement)
    return res


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
        # plt.show(block=True)
        mpim.imsave(os.path.join(save_dir, f"{i}.png"), super_imposed)
        plt.close()


if __name__ == "__main__":
    # mire_info = mire_analysis(constants.MIRE_PATH, visualization=True)
    # mire_info.save(constants.MIRE_INFO_PATH)

    mire_info = MireInfo(constants.MIRE_INFO_PATH)
    # parent_folder = "/Volumes/GUILLAUME/Ficoll Marty/Ficoll17%_20-11-05_1uLbactos_TRACKING"
    # list_dir = [f for f in os.listdir(parent_folder) if not f.startswith(".")]
    # for folder in list_dir[0: 1]:
        # folder_superposition(os.path.join(parent_folder, folder), "/Users/sintes/Desktop", mire_info)
    folder_superposition("/Volumes/GuillaumeHD/SwimmingPVP_23_07_25/2023-07-25_18h12m38s", "/Users/sintes/Desktop", mire_info)