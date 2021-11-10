"""Anlayse the angle between the velocity vector and the flagella."""

import os
from typing import List, Tuple

import numpy as np
import matplotlib.image as mpim
import matplotlib.pyplot as plt

from angleBodyFlagella import detect_flagella
from trackParsing import load_track_data
import superimpose
import constants

def detect_angle(
    super_imposed: np.ndarray,
    visualization: bool = False) ->  float:
    """Detect the angle between the body and the flagella."""
    pass

def list_angle_detection(
    image_list: List[str],
    visualization: bool = False) -> Tuple[List[float], List[float]]:
    """Run the angle detection on a list of path and return angle and time."""
    angle = []
    time = []
    for i, im_path in enumerate(image_list):
        im_test = mpim.imread(im_path) 
        im_test = im_test / np.amax(im_test)
        super_imposed = superimpose.superposition(im_test, mire_info)
        super_imposed = superimpose.select_center_image(super_imposed, 100)
        time.append(i / constants.FPS)
        angle.append(180 * detect_angle(super_imposed, visualization) / np.pi)
    return time, angle


def save_data(time: List[int], angle: List[float]) -> None:
    """Save the data to a text file."""
    textfile = open(os.path.join(constants.FOLDER, "angle_velocity_flagella.csv"), "w")
    for i in range(len(time)):
        textfile.write(f"{time[i]}, {angle[i]}\n")
    textfile.close()

if __name__ == "__main__":
    mire_info = superimpose.MireInfo(constants.MIRE_INFO_PATH)
    image_list = [os.path.join(constants.FOLDER, f) for f in os.listdir(constants.FOLDER) if (f.endswith(".tif") and not f.startswith("."))]

    im_test = mpim.imread(image_list[1508])
    im_test = superimpose.select_center_image(superimpose.superposition(im_test, mire_info), 100)