"""Data analysis for the angle between body and flagella."""

import os
from statistics import mean
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import constants
from trackParsing import load_track_data

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
    return ft_signal

def get_frequencies(signal: List[int], frame_rate: int = constants.FPS ) -> List[float]:
    """Give the frequency list associated with the Fourier transform"""
    freq = np.fft.rfftfreq(len(signal), d=1 / frame_rate)
    return freq

def smooth_angle(angles: List[float], window_time: float) -> List[float]:
    """Smooth the angle on a timescale window_time in secondes by FT cutting."""
    freq = get_frequencies(angles)
    ft_angle = np.fft.rfft(angles)
    i = 0
    f = freq[0]
    while f < 1 / window_time:
        i += 1
        f = freq[i]
    ft_angle[i: ] = 0
    smooth_ang = list(np.fft.irfft(ft_angle))
    smooth_ang.append(0)
    return smooth_ang


def angle_shift(angles: List[float]) -> List[float]:
    """Make the angle betweem -90 and 90 degrees."""
    angles_c = angles.copy()
    for i, ang in enumerate(angles_c):
        if ang < - 90:
            angles_c[i] = ang + 180
        elif ang > 90:
            angles_c[i] = ang - 180
    return angles_c


def detect_extrema(times: List[float], angles: List[float])-> np.ndarray:
    """Detect the extrema of the angles."""
    angles = np.array(angles)
    differences = angles[1: ] - angles[:- 1]
    extrema = []
    for i, diff in enumerate(differences[: - 1]):
        if diff * differences[i + 1] < 0:
            extrema.append((times[i + 1], angles[i + 1]))
    return np.array(extrema)


def get_amplitude(extrema: np.ndarray) -> float:
    """Give the cone angle from the extrema."""
    angles = extrema[:, 1]
    diff = angles[1:] - angles[:-1]
    return np.mean(np.abs(diff)) / 2


def get_period(extrema: np.ndarray) -> float:
    """Give the period from the extrema."""
    times = extrema[:, 0]
    diff = times[1:] - times[:-1]
    return np.mean(np.abs(diff)) * 2


def linear_interpolation(times: List[float], angles: List[float]) -> Tuple[List[float], List[float]]:
    """Make a linear interpolation for the missing data."""
    angle_inter = list(angles.copy())
    times_inter = [k / constants.FPS for k in range(int(min(times) * constants.FPS), int(max(times) * constants.FPS) + 1)]
    diff_times = times[1: ] - times[:len(times) -1]
    for i, diff in enumerate(diff_times):
        if diff > 1.5 / constants.FPS:
            nb_missing = int(diff * constants.FPS) - 1
            slope = (np.mean(angles[i + 1: i + 10]) - np.mean(angles[i - 10: i])) / diff
            for k in range(1, nb_missing + 1):
                ang = angle[i] + k * slope / constants.FPS
                angle_inter.insert(i + k, ang)
    return times_inter, angle_inter


if __name__ == "__main__":
    track_data = load_track_data()
    time, angle = load_data()
    angle = angle_shift(angle)
    
    # time = time[: 480]
    # angle = angle[: 480]
    time, angle = linear_interpolation(time, angle)
    smooth_ang = smooth_angle(angle, 0.5)

    extrema = detect_extrema(time, smooth_ang)

    print("Amplitude mean:", get_amplitude(extrema))
    print("Period:", get_period(extrema))
    plt.figure()
    plt.plot(time, angle, '.')
    plt.plot(time, smooth_ang, "-")
    plt.plot(extrema[:, 0], extrema[:, 1], "*r")
    plt.xlabel("Time (in s)")
    plt.ylabel("Angle (in degrees)")
    
    freq = get_frequencies(smooth_ang)
    ft_angle = fourier_transform(smooth_ang)
    plt.figure()
    plt.plot(get_frequencies(angle), fourier_transform(angle), ".", label="raw")
    plt.plot(freq, ft_angle, label="smooth")
    plt.xlabel("$f\ (in\ s^{-1})$")
    plt.legend()
    print("mean angle:", np.mean(smooth_ang))
    print("Frequency mode:",freq[ft_angle.index(1)])

    plt.show(block=True)