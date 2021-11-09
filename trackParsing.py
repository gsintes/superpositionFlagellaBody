"""Anlayse the angle between the velocity vector and the flagella."""

import os
import re

import numpy as np
import pandas as pd

import constants

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

def calculate_velocities(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the velocities and add them to a new dataframe."""
    data["vel_x"] = 1000 * data["x"].diff() / constants.FPS
    data["vel_x"][0] = 0
    data["vel_y"] = 1000 * data["y"].diff() / constants.FPS
    data["vel_y"][0] = 0
    data["vel_z"] = 1000 * data["z"].diff() / constants.FPS
    data["vel_z"][0] = 0
    for i in range(1, len(data["vel_z"])):
        if i % 2 == 0:
            data["vel_z"][i] = data["vel_z"][i - 1]
    data["vel"] = np.sqrt(data["vel_x"] ** 2 + data["vel_y"] ** 2 + data["vel_z"] ** 2)
    
    return data

if __name__ == "__main__":
    data = parser_track()
    print(calculate_velocities(data))

