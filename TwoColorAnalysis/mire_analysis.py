"""Perform the analysis of the mire image."""

from typing import Tuple
from statistics import median


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpim
from scipy.signal import correlate2d

import constants
import superimpose 
from mire_info import MireInfo


def find_separation(mire_im: np.ndarray, visualization: bool=False) -> int:
    """Open the mire image and find the separation line."""
    loc_profiles = range(10, constants.IM_SIZE[0] - 10, 10)
    separators = []
    for loc_profile in loc_profiles:
        profile = mire_im[:, loc_profile] / max(mire_im[:, loc_profile])
        smooth_prof = superimpose.moving_average(profile, 20)
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


# def find_displacement(
#     green_mire: np.ndarray,
#     red_mire: np.ndarray,
#     visualization: bool = False) -> Tuple[int, int]:
#     """Find the displacement of the two pictures."""

#     cross_corr = correlate2d(
#         binarize(select_center_image(red_mire, 100)),
#         binarize(select_center_image(green_mire, 100)))
    
#     i, j = np.unravel_index(cross_corr.argmax(), cross_corr.shape)
#     delta_x = 1 + i - select_center_image(red_mire, 100).shape[0]
#     delta_y =  1 + j - select_center_image(red_mire, 100).shape[1]

#     if visualization:
#         plt.figure()
#         plt.imshow(cross_corr, cmap="gray")
#         plt.plot(i, j, "*r")
#     return (delta_x, delta_y)

def manual_find_displacement(
    green_mire: np.ndarray,
    red_mire: np.ndarray) -> Tuple[int, int]:
    """Manually find the displacement in the image by clicking on a point."""
    fig = plt.figure()
    center = (red_mire.shape[0] // 2, red_mire.shape[1] // 2)
    plt.imshow(superimpose.select_center_image(green_mire, center, 1000))
    point_one = plt.ginput(1)[0]
    plt.close(fig)
    fig = plt.figure()
    plt.imshow(superimpose.select_center_image(red_mire, center, 200))
    point_two = plt.ginput(1)[0]
    plt.close(fig)

    delta_x = int(point_two[0] - point_one[0])
    delta_y = int(point_two[1] - point_one[1])
    return (delta_y, delta_x)

def mire_analysis(mire_path: str, visualization: bool=True) -> MireInfo:
    """Perform the mire analysis"""
    mire_im = mpim.imread(mire_path) 
    mire_im = mire_im / 2 ** 16
    middle_line = find_separation(mire_im)
    red_mire, green_mire = superimpose.split_image(mire_im, middle_line)

    displacement = manual_find_displacement(green_mire, red_mire)
    res = MireInfo(middle_line, displacement)

    if visualization:
        check_im = superimpose.superposition(mire_im, res)
        plt.figure()
        plt.imshow(check_im * 255)
        plt.show(block=True)
    return res

if __name__ == "__main__":
    mire_info = mire_analysis(constants.MIRE_PATH)
    mire_info.save(constants.MIRE_INFO_PATH)