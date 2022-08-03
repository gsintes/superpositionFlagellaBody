"""Data analysis for the angle between body and flagella."""

import os
import re
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import constants
from trackParsing import load_track_data

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

class Analysis:
    def __init__(self,
        limits: Tuple[int, int],
        folder: str = "",
        data_file: str = constants.ANGLE_DATA_FILE,
        angles: List[float]=[],
        times: List[float]=[]
        ) -> None:
        """Load the angle data."""
        self.folder = folder
        if folder != "":
            data = np.genfromtxt(os.path.join(folder, data_file), delimiter=",")
            time = data[:, 0]
            angle = data[:, 1] 
        else:
            time = times
            angle = angle
        self.limits = limits
        lim_track0 = int(round(min(time) * constants.FPS)) + limits[0]
        lim_track1 = int(round(min(time) * constants.FPS)) + limits[1]
        self.track_data: pd.DataFrame = load_track_data()[lim_track0: lim_track1]
        

        self.times = time
        self.angles = angle.copy()
        self.times = self.times[limits[0]: limits[1]]
        self.angles = angle[limits[0]: limits[1]]
        self.cleaned_angles = self.angles.copy()


    def smooth_angle(self, window_time: float) -> None:
        """Smooth the angle on a timescale window_time in secondes by FT cutting."""
        freq = get_frequencies(self.cleaned_angles)
        ft_angle = np.fft.rfft(self.cleaned_angles)
        i = 0
        f = freq[0]
        while f < 1 / window_time:
            i += 1
            f = freq[i]
        ft_angle[i: ] = 0
        smooth_ang = list(np.fft.irfft(ft_angle))
        if len(self.angles) % 2 == 1:
            smooth_ang.append(0)
        self.cleaned_angles = smooth_ang.copy()


    def angle_shift(self) -> None:
        """Make the angle betweem -90 and 90 degrees."""
        angles_c = self.cleaned_angles.copy()
        for i, ang in enumerate(angles_c):
            if ang < - 90:
                angles_c[i] = ang + 180
            elif ang > 90:
                angles_c[i] = ang - 180
        self.cleaned_angles = angles_c.copy()

    def linear_interpolation(self) -> None:
        """Make a linear interpolation for the missing data."""
        angle_inter = list(self.cleaned_angles.copy())
        times_inter = [k / constants.FPS for k in range(int(min(self.times) * constants.FPS), int(max(self.times) * constants.FPS) + 1)]
        diff_times = self.times[1: ] - self.times[:len(self.times) -1]
        for i, diff in enumerate(diff_times):
            if diff > 1.5 / constants.FPS:
                nb_missing = int(round(diff * constants.FPS)) - 1
                slope = (np.mean(self.cleaned_angles[i + 1: i + 10]) - np.mean(self.cleaned_angles[i - 10: i])) / diff
                for k in range(1, nb_missing + 1):
                    ang = self.cleaned_angles[i] + k * slope / constants.FPS
                    angle_inter.insert(i + k, ang)
        self.times = times_inter.copy()
        self.cleaned_angles = angle_inter

    def clean_data(self, window_size: float) -> None:
        """Prepare the data"""
        self.angle_shift()
        self.linear_interpolation()
        self.smooth_angle(window_size)

    def detect_extrema(self)-> None:
        """Detect the extrema of the angles."""
        angles = np.array(self.cleaned_angles)
        differences = angles[1: ] - angles[:- 1]
        extrema = []
        for i, diff in enumerate(differences[: - 1]):
            if diff * differences[i + 1] < 0:
                extrema.append((self.times[i + 1], angles[i + 1]))
        self.extrema = np.array(extrema)


    def get_amplitude(self) -> None:
        """Give the cone angle from the extrema."""
        angles = self.extrema[:, 1]
        diff = angles[1:] - angles[:-1]
        self.amplitude = np.mean(np.abs(diff)) / 2
        self.std_amplitude = np.std(np.abs(diff)) / 2


    def get_period(self) -> Tuple[float, float]:
        """Give the period from the extrema."""
        times = self.extrema[:, 0]
        diff = times[1:] - times[:-1]
        self.period = np.mean(np.abs(diff)) * 2
        self.std_period = np.std(np.abs(diff)) * 2

    def run_analysis(self, visualization: bool) -> pd.Series:
        """Run the analysis on the section of the data."""
        

        self.clean_data(0.5)
        self.detect_extrema()
        freq = get_frequencies(self.cleaned_angles)
        ft_angle = fourier_transform(self.cleaned_angles)

        self.get_amplitude()
        self.get_period()
        self.mean_angle = np.mean(self.angles)
        self.fourier_mode = freq[ft_angle.index(1)]

        data = pd.Series({
            "Folder": self.folder,
            "Limits": self.limits,
            "Nb_extrema": len(self.extrema),
            "Amplitude": self.amplitude,
            "Std_amplitude": self.std_amplitude,
            "Period": self.period,
            "Std_period": self.std_period,
            "Mean_angle": self.mean_angle,
            "Fourier_mode": self.fourier_mode,
            "Mean_vel": self.track_data["vel"].mean(),
            "Std_vel": self.track_data["vel"].std()
            })

        if visualization:
            plt.figure()
            plt.plot(self.times, self.angles, "-")
            plt.plot(self.extrema[:, 0], self.extrema[:, 1], "*r")
            plt.xlabel("Time (in s)")
            plt.ylabel("Angle (in degrees)")
            plt.savefig(f"{self.folder}/angle.png")

            plt.figure()
            plt.plot(freq, ft_angle, label="smooth")
            plt.xlabel("$f\ (in\ s^{-1})$")
            plt.savefig(f"{self.folder}/fourier.png")
            plt.close()
            # plt.show(block=True)
        return data


def get_limits(lim_lit: str) -> Tuple[int, int]:
    """Tranfroms the string of limits into a tuple of ints."""
    exp = re.search(r"^\((\d*),\ (\d*)", lim_lit)
    lim0 = int(exp.group(1))
    lim1 = int(exp.group(2))
    return (min(lim0, lim1), max(lim0, lim1))


if __name__ == "__main__":
    analysis_data = pd.read_csv(os.path.join(constants.FOLDER_UP, "wobbling_data.csv"))

    data_list: List[pd.Series] = []
    for _, info in analysis_data.iterrows():
        analysis = Analysis(limits=get_limits(info["Limits"]), folder=info["Folder"])
        data_list.append(analysis.run_analysis(visualization=False))

    data = pd.DataFrame()
    data = pd.DataFrame(data_list)
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
