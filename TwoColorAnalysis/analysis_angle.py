"""Data analysis for the angle between body and flagella."""

import os
import re
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def get_frequencies(signal: List[int], frame_rate: int ) -> List[float]:
    """Give the frequency list associated with the Fourier transform"""
    freq = np.fft.rfftfreq(len(signal), d=1 / frame_rate)
    return freq

class Analysis:
    count_data = 0
    def __init__(self,
        limits: Tuple[int, int],
        fps: int,
        folder: str = "",
        angles: List[float]=[],
        times: List[float]=[]
        ) -> None:
        """Load the angle data."""
        data_file = "angle_body_flagella.csv"
        self.fps = fps
        Analysis.count_data += 1
        self.folder = folder
        if folder != "":
            data = pd.read_csv(os.path.join(folder, data_file), delimiter=",")
            time = data["Time"] * 80 / self.fps #TODO remove when all code re runned
            angle = data[" FlagellaBody angle"]
        else:
            time = times
            angle = angles
        self.limits = limits
        lim_track0 = int(round(min(time) * self.fps)) + limits[0]
        lim_track1 = int(round(min(time) * self.fps)) + limits[1]
        self.track_data: pd.DataFrame = load_track_data(folder=folder, fps=self.fps)[lim_track0: lim_track1]

        self.window_size = 0.25
        self.times = time
        self.angles = angle.copy()
        self.times = self.times[limits[0]: limits[1]]
        self.angles = angle[limits[0]: limits[1]]
        self.cleaned_angles = self.angles.copy()


    def _smooth_angle(self) -> None:
        """Smooth the angle on a timescale window_time in secondes by FT cutting."""
        ft_angle = np.fft.rfft(self.cleaned_angles)
        spectrum = np.abs(ft_angle)

        for i, _ in enumerate(ft_angle):
            if spectrum[i] < 0.15 * max(spectrum):
                ft_angle[i] = 0
        smooth_ang = list(np.fft.irfft(ft_angle))
        if len(self.cleaned_angles) % 2 == 1:
            smooth_ang.append(0)
        self.cleaned_angles = smooth_ang.copy()


    def _angle_shift(self) -> None:
        """Make the angle betweem -90 and 90 degrees."""
        angles_c = self.cleaned_angles.copy()
        for i, ang in enumerate(angles_c):
            if ang < - 90:
                angles_c[i] = ang + 180
            elif ang > 90:
                angles_c[i] = ang - 180
        self.cleaned_angles = angles_c.copy()

    def _linear_interpolation(self) -> None: #FIXME
        """Make a linear interpolation for the missing data."""
        angle_inter = list(self.cleaned_angles.copy())
        times_inter = [k / self.fps for k in range(int(min(self.times) * self.fps), int(max(self.times) * self.fps) + 1)]
        diff_times = self.times[1: ] - self.times[:len(self.times) -1]
        for i, diff in enumerate(diff_times):
            if diff > 1.5 / self.fps:
                nb_missing = int(round(diff * self.fps)) - 1
                slope = (np.mean(self.cleaned_angles[i + 1: i + 10]) - np.mean(self.cleaned_angles[i - 10: i])) / diff
                for k in range(1, nb_missing + 1):
                    ang = self.cleaned_angles[i] + k * slope / self.fps
                    angle_inter.insert(i + k, ang)
        self.cleaned_times = times_inter.copy()
        self.cleaned_angles = angle_inter
        print("Time", len(self.times), "Cleaned times", len(self.cleaned_times))
        print(len(self.cleaned_times) - len(self.times))
        print("Angle", len(self.angles), "Cleaned_angles", len(self.cleaned_angles))
        print(len(self.cleaned_angles) - len(self.angles))

    def _clean_data(self) -> None:
        """
        Prepare the data.

        window_size: float : The timestep (in s) for smoothing.
        """
        self._angle_shift()
        self._linear_interpolation()
        self._smooth_angle()


    def _detect_extrema(self)-> None:
        """Detect the extrema of the angles."""
        angles = np.array(self.cleaned_angles)
        differences = angles[1: ] - angles[:- 1]
        extrema = []
        for i, diff in enumerate(differences[: -1]):
            if diff * differences[i + 1] < 0:
                try:
                    extrema.append((self.cleaned_times[i + 1], angles[i + 1]))
                except IndexError: #FIXME
                    pass
        extrema_a = np.array(extrema)


        diff = extrema_a[1:, 0] - extrema_a[:-1, 0]
        th_diff = 0.6 * np.mean(diff)
        count = 0
        for i in range(len(diff)):
            if diff[i] < th_diff:
                count += 1
                extrema.pop(i + 1 - count)
        self.extrema = np.array(extrema)

    def _get_amplitude(self) -> None:
        """Give the cone angle from the extrema."""
        angles = self.extrema[:, 1]
        diff = angles[1:] - angles[:-1]
        self.amplitude = np.mean(np.abs(diff)) / 2
        self.std_amplitude = np.std(np.abs(diff)) / 2


    def _get_period(self) -> Tuple[float, float]:
        """Give the period from the extrema."""
        times = self.extrema[:, 0]
        diff = times[1:] - times[:-1]
        self.period = np.mean(np.abs(diff)) * 2
        self.std_period = np.std(np.abs(diff)) * 2

    def __call__(self, visualization: bool) -> pd.Series:
        """Run the analysis on the section of the data."""
        self._clean_data()
        self._detect_extrema()
        freq = get_frequencies(self.cleaned_angles, self.fps)
        ft_angle = fourier_transform(self.cleaned_angles)

        self._get_amplitude()
        self._get_period()
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
            plt.plot(self.times, self.angles, "-", label="raw")
            # plt.plot(self.cleaned_times, self.cleaned_angles, "-", label="Smooth") #TODO Fixme
            plt.plot(self.extrema[:, 0], self.extrema[:, 1], "*r")
            plt.xlabel("Time (in s)")
            plt.ylabel("Angle (in degrees)")
            plt.legend()
            plt.savefig(f"{constants.FOLDER_UP}/Wobbling/angle_{Analysis.count_data}.png")#TODO change naming
            plt.close()

            plt.figure()
            ft_angle = fourier_transform(self.angles)
            spectrum = np.abs(ft_angle) / max(np.abs(ft_angle))
            plt.plot(get_frequencies(self.angles, self.fps), spectrum, label="Raw")
            ft_angle = fourier_transform(self.cleaned_angles)
            spectrum = np.abs(ft_angle) / max(np.abs(ft_angle))
            plt.plot(freq, ft_angle, label="smooth")
            plt.xlabel("$f\ (in\ s^{-1})$")
            plt.legend()
            plt.savefig(f"{constants.FOLDER_UP}/Wobbling/fourier_{Analysis.count_data}.png")
            plt.close()
            # plt.show(block=True)
        return data


def get_limits(lim_lit: str) -> Tuple[int, int]:
    """Tranfroms the string of limits into a tuple of ints."""
    exp = re.search(r"^(\d*)-(\d*)", lim_lit)
    lim0 = int(exp.group(1))
    lim1 = int(exp.group(2))
    return (min(lim0, lim1), max(lim0, lim1))


if __name__ == "__main__":
    folder_up = ""
    analysis_data = pd.read_csv(os.path.join(folder_up, "exp-info.csv"))

    data_list: List[pd.Series] = []
    for _, info in analysis_data.iterrows():
        limits = info["limits"]
        concentration = info["concentration"]
        viscosity = info["viscosity"]
        if info["final_flagella_frame"] != 0 and type(limits) == str:
            limit_list = info["limits"].split("/")
            folder = info["exp"]
            fps = info["fps"]
            if len(folder.split("/")) == 1:
                folder = os.path.join(folder_up, folder)
            for lim in limit_list:
                if lim != "":

                    analysis = Analysis(limits=get_limits(lim), folder=folder, fps=fps)
                    exp_data = analysis(visualization=True)
                    exp_data["Concentration"] = concentration
                    exp_data["Viscosity"] = viscosity
                    data_list.append(exp_data)

    data = pd.DataFrame()
    data = pd.DataFrame(data_list)
    data.to_csv(os.path.join(folder_up, "wobbling_data.csv"))

    data["Count_freq"] = 1 / data["Period"]
    x = np.linspace(min(data["Fourier_mode"]), max(data["Fourier_mode"]))
    plt.figure()
    plt.plot(data["Fourier_mode"], data["Count_freq"], ".")
    plt.plot(x, x, "k--")
    plt.xlabel("Fourier frequency (Hz)")
    plt.ylabel("Count frequency (Hz)")
    plt.savefig(f"{folder_up}/Wobbling/freq_freq.png")

    data.plot("Amplitude", "Mean_angle", "scatter")
    x = np.linspace(min(data["Amplitude"]), max(data["Amplitude"]))
    plt.plot(x, x, "k--")
    plt.plot(x, -x, "k--")
    plt.savefig(f"{folder_up}/Wobbling/mean_amplitude.png")


    data.plot("Mean_vel", "Period", "scatter")
    plt.savefig(f"{folder_up}/Wobbling/period_vel.png")
    data.plot("Mean_vel", "Amplitude", "scatter")
    plt.savefig(f"{folder_up}/Wobbling/amplitude_vel.png")
    data.plot("Period", "Amplitude", "scatter")
    plt.savefig(f"{folder_up}/Wobbling/amplitude_period.png")
    plt.show(block=True)
