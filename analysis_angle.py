"""Data analysis for the angle between body and flagella."""

import os
import re
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.shape_base import block
import pandas as pd

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
    if len(angles) % 2 == 1:
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


def get_amplitude(extrema: np.ndarray) -> Tuple[float, float]:
    """Give the cone angle from the extrema."""
    angles = extrema[:, 1]
    diff = angles[1:] - angles[:-1]
    return np.mean(np.abs(diff)) / 2, np.std(np.abs(diff)) / 2


def get_period(extrema: np.ndarray) -> Tuple[float, float]:
    """Give the period from the extrema."""
    times = extrema[:, 0]
    diff = times[1:] - times[:-1]
    return np.mean(np.abs(diff)) * 2, np.std(np.abs(diff)) * 2


def linear_interpolation(times: List[float], angles: List[float]) -> Tuple[List[float], List[float]]:
    """Make a linear interpolation for the missing data."""
    angle_inter = list(angles.copy())
    times_inter = [k / constants.FPS for k in range(int(min(times) * constants.FPS), int(max(times) * constants.FPS) + 1)]
    diff_times = times[1: ] - times[:len(times) -1]
    for i, diff in enumerate(diff_times):
        if diff > 1.5 / constants.FPS:
            nb_missing = int(round(diff * constants.FPS)) - 1
            slope = (np.mean(angles[i + 1: i + 10]) - np.mean(angles[i - 10: i])) / diff
            for k in range(1, nb_missing + 1):
                ang = angles[i] + k * slope / constants.FPS
                angle_inter.insert(i + k, ang)
    return times_inter, angle_inter


def clean_data(time: List[float], angle: List[float], window_size: float) -> Tuple[List[float], List[float]]:
    """Prepare the data"""
    angle = angle_shift(angle)
    time, angle = linear_interpolation(time, angle)
    smooth_ang = smooth_angle(angle, window_size)
    return  time, smooth_ang


def run_analysis(folder: str, limits: Tuple[int, int], visualization: bool) -> pd.Series:
    """Run the analysis on the section of the data."""
    time, angle = load_data(folder)
    lim_track0 = int(round(min(time) * constants.FPS)) + limits[0]
    lim_track1 = int(round(min(time) * constants.FPS)) + limits[1]
    track_data: pd.DataFrame = load_track_data()[lim_track0: lim_track1]
    time = time[limits[0]: limits[1]]
    angle = angle[limits[0]: limits[1]]
    time, angle = clean_data(time, angle, 0.5)
    extrema = detect_extrema(time, angle)
    freq = get_frequencies(angle)
    ft_angle = fourier_transform(angle)

    amplitude, std_a = get_amplitude(extrema)
    period, std_p = get_period(extrema)
    mean_angle = np.mean(angle)
    fourier_mode = freq[ft_angle.index(1)]

    data = pd.Series({
        "Folder": folder,
        "Limits": limits,
        "Nb_extrema": len(extrema),
        "Amplitude": amplitude,
        "Std_amplitude": std_a,
        "Period": period,
        "Std_period": std_p,
        "Mean_angle": mean_angle,
        "Fourier_mode": fourier_mode,
        "Mean_vel": track_data["vel"].mean(),
        "Std_vel": track_data["vel"].std()
        })

    if visualization:
        plt.figure()
        plt.plot(time, angle, "-")
        plt.plot(extrema[:, 0], extrema[:, 1], "*r")
        plt.xlabel("Time (in s)")
        plt.ylabel("Angle (in degrees)")
        plt.savefig(f"{folder}/angle.png")

        plt.figure()
        plt.plot(freq, ft_angle, label="smooth")
        plt.xlabel("$f\ (in\ s^{-1})$")
        plt.savefig(f"{folder}/fourier.png")
        plt.close()
        # plt.show(block=True)
    return data


def get_limits(lim_lit: str) -> Tuple[int, int]:
    """Tranfroms the string of limits into a tuple of ints."""
    exp = re.search(r"^\((\d*),\ (\d*)", lim_lit)
    lim0 = int(exp.group(1))
    lim1 = int(exp.group(2))
    return (lim0, lim1)


if __name__ == "__main__":
    analysis_data = pd.read_csv(os.path.join(constants.FOLDER_UP, "wobbling_data.csv"))
    data = pd.DataFrame()
    data = pd.DataFrame(
        [run_analysis(info["Folder"], get_limits(info["Limits"]), False) for _, info in analysis_data.iterrows()]
        )
    data.to_csv(os.path.join(constants.FOLDER_UP, "wobbling_data.csv"))
    
    data["Count_freq"] = 1 / data["Period"]
    x = np.linspace(min(data["Fourier_mode"]), max(data["Fourier_mode"]))
    plt.figure()
    plt.plot(data["Fourier_mode"], data["Count_freq"], ".")
    plt.plot(x, x, "k--")
    plt.xlabel("Fourier frequency (Hz)")
    plt.ylabel("Count frequency (Hz)")

    data.plot("Mean_vel", "Period", "scatter")
    data.plot("Mean_vel", "Amplitude", "scatter")
    data.plot("Period", "Amplitude", "scatter")
    plt.show(block=True)
