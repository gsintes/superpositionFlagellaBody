"""Anlayse the angle between the velocity vector and the flagella."""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.image as mpim
import matplotlib.pyplot as plt
from pandas.io.parquet import FastParquetImpl

from angleBodyFlagella import detect_flagella
from trackParsing import load_track_data
import superimpose
import constants

def detect_angle(
    vel_param: pd.Series,
    super_imposed: np.ndarray,
    i,
    visualization: bool = False) ->  float:
    """Detect the angle between the velocity and the flagella."""
    vel_x = vel_param["vel_x"]
    vel_y = vel_param["vel_y"]
    a0 = vel_param["slope"]
    b0 = vel_param["b_coeff"]

    a1, b1, vect = detect_flagella(super_imposed[:, :, 0], visualization=False)
    x = np.linspace(0, super_imposed.shape[0])
    scalar_prod = (vect[0] * vel_x) + (vect[1] * vel_y)
    norm_vel = (vel_y ** 2 + vel_x ** 2) ** (1/2)
    if norm_vel > 0:
        theta = np.arccos(scalar_prod / norm_vel) - 3 * np.pi / 4
    else:
        theta = 0
        
    if visualization:
        plt.imshow(super_imposed)
        plt.plot(a0 * x + b0, x, "-b", linewidth=3)
        plt.plot(a1 * x + b1, x, "-r", linewidth=3)
        plt.ylim([super_imposed.shape[0], 0])
        plt.xlim([0, super_imposed.shape[1]])
        plt.draw()
        plt.pause(0.001)
        plt.savefig(f"/Users/sintes/Desktop/VideoAngleVel/im{i}.png")
        plt.clf()
        plt.close()
    
    return theta

def list_angle_detection(
    track_data: pd.DataFrame,
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

        angle.append(180 * detect_angle(
            track_data.iloc[i],
            super_imposed,
            i,
            visualization) / np.pi)
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
    track_data = load_track_data()
    image_list = image_list[:track_data.shape[0]]

    time, angle = list_angle_detection(track_data, image_list, visualization=True)
    save_data(time, angle)
    
    plt.figure()
    plt.plot(time, angle)
    plt.xlabel("Time (in s)")
    plt.ylabel("$\phi$ (in $\degree$)")
    plt.savefig(os.path.join(constants.FOLDER, "angle_velocity_flagella.png"))
    plt.show(block=True)
