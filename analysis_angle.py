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


def fourier_transform(signal: List[float]) -> List[float]:
    """
    Calculate the Fourier transform of the signal.

    Input :
    signal: List[float]

    Output:
    Fourier transform: List[float]
    """
    ft_signal = np.abs(np.fft.rfft(signal))
    ft_signal = ft_signal[1:] / max(ft_signal[1:])
    return ft_signal


def get_frequencies(signal: List[int], frame_rate: int = constants.FPS ) -> List[float]:
    """Give the frequency list associated with the Fourier transform"""
    freq = np.array([frame_rate/ (2 * i) for i in range(1, len(signal) // 2 + 2)])
    freq = freq[1:]
    return freq


if __name__ == "__main__":
    time, angle = load_data()
    
    # time = time[1507: 1572]
    # angle = angle[1507: 1572]

    plt.figure()
    plt.plot(time, angle, ".")
    plt.xlabel("Time (in s)")
    plt.ylabel("Angle (in degrees)")
    

    plt.figure()
    plt.plot(get_frequencies(angle), fourier_transform(angle), ".")
    plt.show(block=True)