"""Parse the track data."""

import os
import re
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import constants
from superimpose import MireInfo


def parser_track(
    folder: str = constants.FOLDER,
    file: str = constants.TRACK_FILE) -> pd.DataFrame:
    """Parse the data from the track file"""
    exp = r"^\s*\d+\s+(-*\d+.\d+)\s+(-*\d+.\d+)\s+(-*\d+.\d+)\s+-*\d+.\d+\s+-*\d+.\d+\s+-*\d+.\d+\s+-*\d+.\d+\s+-*\d+.\d+\s+-*\d+.\d+\s+(\d+.\d+)\s+(\d+.\d+)"
    x = []
    y = []
    z = []
    time = []
    center_x = []
    center_y = []
    with open(os.path.join(folder, file), "r") as f:
        for i, line in enumerate(f):
            expression = re.search(exp, line)
            x.append(float(expression.group(1)))
            y.append(float(expression.group(2)))
            z.append(float(expression.group(3)))
            center_x.append(float(expression.group(4)))
            center_y.append(float(expression.group(5)))
            time.append(i / constants.FPS) 
    data = pd.DataFrame()
    data["time"] = time
    data["x"] = x
    data["y"] = y
    data["z"] = z
    data["center_x"] = center_x
    data["center_y"] = center_y
    return data


def smooth_trajectory(data: pd.DataFrame, window_size:int) -> pd.DataFrame:
    """Smooth the trajectories."""
    data["smooth_x"] = data['x'].rolling(window=window_size).mean()
    data["smooth_y"] = data['y'].rolling(window=window_size).mean()
    data["smooth_z"] = data['z'].rolling(window=window_size).mean()
    return data


def smooth_derivative(vector: Iterable, step_size: float) -> Iterable:
    """Does a smooth derivative of the data."""
    derivative = [0, 0]
    for i in range(2, len(vector) - 2):
        der = (2 * (vector[i + 1] - vector[i - 1]) + vector[i +2] - vector[i - 2]) / (8 * step_size)
        derivative.append(der)
    derivative.append(0)
    derivative.append(0)
    return np.array(derivative)


def calculate_velocities(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the velocities and add them to a new dataframe."""
    data["vel_x"] = smooth_derivative(data["smooth_x"], 1 / constants.FPS)
    data["vel_y"] = smooth_derivative(data["smooth_y"], 1 / constants.FPS)
    data["vel_z"] = smooth_derivative(data["smooth_z"], 1 / constants.FPS)

    data["vel"] = np.sqrt(data["vel_x"] ** 2 + data["vel_y"] ** 2 + data["vel_z"] ** 2)
    data["slope"] = - data["vel_x"] / data["vel_y"]
    mire_info = MireInfo(constants.MIRE_INFO_PATH)
    
    shift = - mire_info.middle_line - (constants.IM_SIZE[1] - mire_info.middle_line) / 2 + 100
    data["b_coeff"] = - ( data["center_y"] + shift + mire_info.displacement[1]) * data["slope"] +\
         (data["center_x"] + mire_info.displacement[0]-  (constants.IM_SIZE[1] / 2) + 100)
    return data


def load_track_data(
    folder: str = constants.FOLDER,
    file: str = constants.TRACK_FILE) -> pd.DataFrame:
    """Load, parse and calculate velocities from track data."""
    data = parser_track(folder, file)
    data = smooth_trajectory(data, 40)
    return calculate_velocities(data)


if __name__ == "__main__":
    data = load_track_data()

    data.plot("time", ["vel_x", "vel_y"])
    # data.plot("time", "x")
    data.plot("time", "y")
    folder = constants.FOLDER.split("/")[-1]
    data.plot("time", ["center_x", "center_y"])
    # plt.savefig(f"/Users/sintes/Desktop/yTracking/{folder}")
    data.plot("time", "smooth_y")
    plt.show(block=True)
