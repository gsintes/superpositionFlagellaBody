"""Data analysis for the angle."""

import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import constants

def load_data(
    folder: str = constants.FOLDER,
    data_file: str = constants.ANGLE_DATA_FILE) -> Tuple[List[float], List[float]]:
    """Load the angle data."""
    data = np.genfromtxt(os.path.join(folder, data_file), delimiter=",")
    time = data[:, 0]
    angle = data[:, 1] 

    return time, angle


if __name__ == "__main__":
    time, angle = load_data()

    plt.figure()
    plt.plot(time, angle, ".", markersize=0.5)
    plt.xlabel("Time (in s)")
    plt.ylabel("Angle (in degrees)")
    plt.show(block=True)