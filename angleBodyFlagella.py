"""Mesure the angle between the body and the flagella along time."""

import os
from typing import Tuple, List

import matplotlib.image as mpim
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian

import superimpose
import constants


def detect_body(
    green_im: np.ndarray,
    visualization: bool = False) -> Tuple[float, float]:
    """Detect the body in the green image."""
    blur = gaussian(green_im, 2)
    bin_green = superimpose.binarize(blur)
    x = []
    y = []
    for i in range(bin_green.shape[0]):
        for j in range(bin_green.shape[1]):
            if bin_green[i, j] == 1:
                x.append(i)
                y.append(j)
    x = np.array(x)
    y = np.array(y)
    a, b =np.polyfit(x, y, 1)
    if visualization:
        fig, axis =plt.subplots(1, 2)
        plt.suptitle("Body detection")
        axis[0].imshow(green_im, cmap="gray")
        axis[0].plot(a * x + b, x, "-g", linewidth=3)
        axis[1].imshow(bin_green, cmap="gray")
        axis[1].plot(a * x + b, x, "-g", linewidth=3)
    return a, b

def detect_flagella(
    red_im: np.ndarray,
    visualization: bool = False) -> Tuple[float, float]:
    """Detect the flagella in the red image."""
    blur = gaussian(red_im, 2)
    bin_red = superimpose.binarize(blur)
    x = []
    y = []
    for i in range(bin_red.shape[0]):
        for j in range(bin_red.shape[1]):
            if bin_red[i, j] == 1:
                x.append(i)
                y.append(j)
    x = np.array(x)
    y = np.array(y)
    a, b =np.polyfit(x, y, 1)
    if visualization:
        _, axis =plt.subplots(1, 2)
        plt.suptitle("Flagella detection")
        axis[0].imshow(red_im, cmap="gray")
        axis[0].plot(a * x + b, x, color="r", linewidth=3)
        axis[1].imshow(bin_red, cmap="gray")
        axis[1].plot(a * x + b, x, color="r", linewidth=3)
    return a, b

def detect_angle(
    super_imposed: np.ndarray,
    visualization: bool = False) ->  float:
    """Detect the angle between the body and the flagella."""
    a0, b0 = detect_body(super_imposed[:, :, 1], visualization)
    a1, b1 = detect_flagella(super_imposed[:, :, 0], visualization)
    x = np.linspace(0, super_imposed.shape[0])
    if visualization:
        plt.figure()
        plt.imshow(super_imposed)
        plt.plot(a0 * x + b0, x, "-g", linewidth=3)
        plt.plot(a1 * x + b1, x, "-r", linewidth=3)
    return np.arctan(a1) - np.arctan(a0)

def save_data(time: List[int], angle: List[float]) -> None:
    """Save the data to a text file."""
    textfile = open(os.path.join(constants.FOLDER, "angle.csv"), "w")
    for i in range(len(time)):
        textfile.write(f"{time[i]}, {angle[i]}\n")
    textfile.close()

def list_angle_detection(image_list: List[str]) -> Tuple[List[float], List[float]]:
    """Run the angle detection on a list of path and return angle and time."""
    angle = []
    time = []
    for i, im_path in enumerate(image_list):
        im_test = mpim.imread(im_path) 
        im_test = im_test / np.amax(im_test)
        super_imposed = superimpose.superposition(im_test, mire_info)
        super_imposed = superimpose.select_center_image(super_imposed, 100)
        time.append(i / constants.FPS)
        angle.append(180 * detect_angle(super_imposed, visualization=False) / np.pi)
    return time, angle


if __name__ == "__main__":
    mire_info = superimpose.MireInfo(constants.MIRE_INFO_PATH)
    image_list = [os.path.join(constants.FOLDER, f) for f in os.listdir(constants.FOLDER) if (f.endswith(".tif") and not f.startswith("."))]
    # sub_list_images = image_list[1507: 1572]
    
    time, angle = list_angle_detection(image_list)    
    save_data(time, angle)

    plt.figure()
    plt.plot(time, angle, ".")
    plt.show(block=True)
