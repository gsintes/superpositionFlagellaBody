"""Analysis of the angle between the velocity and the flagella."""

import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import constants
import analysis_angle

def load_data(
    folder: str = constants.FOLDER,
    data_file: str = "angle_velocity_flagella.csv") -> Tuple[List[float], List[float]]:
    """Load the angle data."""
    data = np.genfromtxt(os.path.join(folder, data_file), delimiter=",")
    time = data[:, 0]
    angle = data[:, 1] 

    return time, angle

if __name__ == "__main__":
    time, angle = load_data()
    print(angle)
    print(np.mean(angle[2: -2]), np.std(angle[2: -2]))

    plt.figure()
    plt.plot(time, angle)
    plt.xlabel("Time (in s)")
    plt.ylabel("Angle (in $\degree$)")
    

    plt.figure()
    plt.plot(analysis_angle.get_frequencies(angle), analysis_angle.fourier_transform(angle))
    plt.xlabel("Frequency (in Hz)")
    plt.ylabel("Fourier transform")
    plt.show(block=True)
