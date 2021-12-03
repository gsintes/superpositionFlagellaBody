"""Data analysis for the angle between body and flagella."""

import os
from statistics import mean
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
    ft_signal = np.abs(np.fft.rfft(signal - np.mean(signal)))
    ft_signal = list(ft_signal / max(ft_signal))
    ft_signal.reverse()
    return ft_signal

def get_frequencies(signal: List[int], frame_rate: int = constants.FPS ) -> List[float]:
    """Give the frequency list associated with the Fourier transform"""
    freq = np.array([frame_rate/ (2 * i) for i in range(1, len(signal) // 2 + 2)])
    return freq

def smooth_angle(times: List[float], angles: List[float], window_time: float) -> List[float]:
    """Smooth the angle on a timescale window_time in secondes."""
    smooth_angle = angles.copy()
    for i, _ in enumerate(angles):
        t1 = times[i]
        t0 = t1 - window_time
        indexes = []
        for k in range(int(i - (window_time * constants.FPS)), i + 1):
            if times[k] >= t0 and times[k] <= t1:
                indexes.append(k)
        smooth_angle[i] = mean([angles[ind] for ind in indexes])
    return smooth_angle


def angle_shift(angles: List[float]) -> List[float]:
    """Make the angle betweem -90 and 90 degrees."""
    angles_c = angles.copy()
    for i, ang in enumerate(angles_c):
        if ang < - 90:
            angles_c[i] = ang + 180
        if ang > 90:
            angles_c[i] = ang - 180
    
    return angles_c

if __name__ == "__main__":
    time, angle = load_data()
    angle = angle_shift(angle)
    print(np.mean(angle))
    # time = time[160: 480]
    # angle = angle[160: 480]
    smooth_ang = smooth_angle(time, angle, 0.25)
    plt.figure()
    plt.plot(time, angle, ".", label="raw")
    plt.plot(time, smooth_ang, "-", label="smooth")
    plt.xlabel("Time (in s)")
    plt.legend()
    plt.ylabel("Angle (in degrees)")
    

    # plt.figure()
    # plt.plot(get_frequencies(angle[:160]), fourier_transform(angle[:160]), ".", label="raw")
    # plt.plot(get_frequencies(smooth_ang[:160]), fourier_transform(smooth_ang[:160]), label="smooth")
    # plt.xlabel("$f\ (in\ s^{-1})$")
    # plt.legend()

    plt.show(block=True)