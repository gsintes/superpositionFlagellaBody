"""Perform the analysis of the mire image."""

from typing import Tuple
import json
from statistics import median


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpim
from scipy.signal import correlate2d

import constants
import superpositionTools as st


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



def find_separation(mire_im: np.ndarray, visualization: bool=False) -> int:
    """Open the mire image and find the separation line."""
    loc_profiles = range(10, constants.IM_SIZE[0] - 10, 10)
    separators = []
    for loc_profile in loc_profiles:
        profile = mire_im[:, loc_profile] / max(mire_im[:, loc_profile])
        smooth_prof = st.moving_average(profile, 20)
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
    red_mire) -> Tuple[int, int]:
    """Manually find the displacement in the image by clicking on a point."""
    fig = plt.figure()
    plt.imshow(st.select_center_image(green_mire, 100))
    point_one = plt.ginput(1)[0]
    plt.close(fig)
    fig = plt.figure()
    plt.imshow(st.select_center_image(red_mire, 100))
    point_two = plt.ginput(1)[0]
    plt.close(fig)

    delta_x = int(point_two[0] - point_one[0])
    delta_y = int(point_two[1] - point_one[1])
    return (delta_y, delta_x)


def mire_analysis(mire_path: str) -> MireInfo:
    """Perform the mire analysis"""
    mire_im = mpim.imread(mire_path) 
    mire_im = mire_im / 2 ** 16
    middle_line = find_separation(mire_im)
    red_mire, green_mire = st.split_image(mire_im, middle_line)

    displacement = manual_find_displacement(green_mire, red_mire)
    res = MireInfo(middle_line, displacement)
    return res

if __name__ == "__main__":
    mire_info = mire_analysis(constants.MIRE_PATH)
    mire_info.save(constants.MIRE_INFO_PATH)