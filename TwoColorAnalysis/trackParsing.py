"""Parse the track data."""

import os
import re
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from superimpose import MireInfo


def parser_track(
    folder: str ,
    fps: int = 80) -> pd.DataFrame:
    """Parse the data from the track file"""
    exp = r"^\s*\d+\s+(-*\d+.\d+)\s+(-*\d+.\d+)\s+(-*\d+.\d+)\s+-*\d+.\d+\s+-*\d+.\d+\s+-*\d+.\d+\s+-*\d+.\d+\s+-*\d+.\d+\s+-*\d+.\d+\s+(\d+.\d+)\s+(\d+.\d+)"
    x = []
    y = []
    z = []
    time = []
    center_x = []
    center_y = []
    with open(os.path.join(folder, "Track/Track.txt"), "r") as f:
        for i, line in enumerate(f):
            expression = re.search(exp, line)
            x.append(float(expression.group(1)))
            y.append(float(expression.group(2)))
            z.append(float(expression.group(3)))
            center_x.append(float(expression.group(4)))
            center_y.append(float(expression.group(5)))
            time.append(i / fps)
    data = pd.DataFrame()
    data["time"] = time
    data["x"] = x
    data["y"] = y
    data["z"] = z
    data["center_x"] = center_y
    data["center_y"] = center_x
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


def calculate_velocities(data: pd.DataFrame, fps: int) -> pd.DataFrame:
    """Calculate the velocities and add them to a new dataframe."""
    data["vel_x"] = smooth_derivative(data["smooth_x"], 1 / fps)
    data["vel_y"] = smooth_derivative(data["smooth_y"], 1 / fps)
    data["vel_z"] = smooth_derivative(data["smooth_z"], 1 / fps)

    data["vel"] = np.sqrt(data["vel_x"] ** 2 + data["vel_y"] ** 2 + data["vel_z"] ** 2)
    return data


def load_track_data(
    folder: str,
    fps: float = 80) -> pd.DataFrame:
    """Load, parse and calculate velocities from track data."""
    data = parser_track(folder, fps)
    data = smooth_trajectory(data, 40) #TODO find intelligent way
    return calculate_velocities(data, fps)

def load_info_exp(file:str, exp: str) -> pd.DataFrame:
    """Load the info on the experiment."""
    info = pd.read_csv(file)
    info_exp = info.loc[info["exp"]==exp]
    return info_exp
