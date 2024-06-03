"""Run the final plotting."""

import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_data(file: str) -> pd.DataFrame:
    """Load the data."""
    return pd.read_csv(file)

def plots(data: pd.DataFrame, filter: str) -> None:
    """Run all the plots"""

    dataLC = data[data["Fluid"] == "LC"]
    dataFicoll = data[data["Fluid"] == "Ficoll"]

    x = np.linspace(min(data["Fourier_mode"]), max(data["Fourier_mode"]))
    sns.relplot(x="Fourier_mode", y="Count_freq", hue="Fluid", style="Fluid", data=data)
    plt.plot(x, x, "k-")
    plt.plot(x, 0.75 * x, "k--")
    plt.plot(x, 1.25 * x, "k--")
    plt.savefig(os.path.join(fig_folder, f"frequency_check_{filter}.png"))

    plt.figure()
    plt.errorbar(x=dataLC["Period"], xerr=dataLC["Std_period"], y=dataLC["Mean_angle"], linestyle="", marker="o", label="LC")
    plt.errorbar(x=dataFicoll["Period"], xerr=dataFicoll["Std_period"], y=dataFicoll["Mean_angle"], linestyle="", marker="*", label="Ficoll")
    plt.legend()
    plt.xlabel("Period (s)")
    plt.ylabel("Mean_angle ($\degree$)")
    plt.savefig(os.path.join(fig_folder, f"period_mean_angle_{filter}.png"))

    plt.figure()
    plt.errorbar(x=dataLC["Mean_vel"], y=dataLC["Period"], yerr=dataLC["Std_period"], linestyle="", marker="o", label="LC")
    plt.errorbar(x=dataFicoll["Mean_vel"], y=dataFicoll["Period"], yerr=dataFicoll["Std_period"], linestyle="", marker="*", label="Ficoll")
    plt.legend()
    plt.xlabel("Velocity (um/s)")
    plt.ylabel("Period (s)")
    plt.savefig(os.path.join(fig_folder, f"period_vel_{filter}.png"))

    plt.figure()
    plt.errorbar(x=dataLC["Mean_vel"], y=dataLC["Amplitude"], yerr=dataLC["Std_amplitude"], linestyle="", marker="o", label="LC")
    plt.errorbar(x=dataFicoll["Mean_vel"], y=dataFicoll["Amplitude"], yerr=dataFicoll["Std_amplitude"], linestyle="", marker="*", label="Ficoll")
    plt.legend()
    plt.xlabel("Velocity (um/s)")
    plt.ylabel("Amplitude ($\degree$)")
    plt.savefig(os.path.join(fig_folder, f"amplitude_vel_{filter}.png"))

    plt.figure()
    plt.errorbar(y=dataLC["Mean_angle"], x=dataLC["Amplitude"], xerr=dataLC["Std_amplitude"], linestyle="", marker="o", label="LC")
    plt.errorbar(y=dataFicoll["Mean_angle"], x=dataFicoll["Amplitude"], xerr=dataFicoll["Std_amplitude"], linestyle="", marker="*", label="Ficoll")
    plt.legend()
    plt.ylabel("Mean angle ($\degree$)")
    plt.xlabel("Amplitude ($\degree$)")
    x = np.linspace(min(data["Amplitude"]), max(data["Amplitude"]))
    plt.plot(x, x, "k-")
    plt.plot(x, -x, "k-")
    plt.savefig(os.path.join(fig_folder, f"amplitude_mean_angle_{filter}.png"))

    sns.relplot(x="Period", y="Amplitude", hue="Fluid", style="Fluid", data=data)
    plt.savefig(os.path.join(fig_folder, f"amplitude_period_{filter}.png"))

def filtering_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filter the data on the comparison of the 2 frequencies, and mean angle."""
    data = data[0.75 * data["Count_freq"] < data["Fourier_mode"]]
    data = data[data["Fourier_mode"] < 1.25 * data["Count_freq"]]
    data = data[data["Mean_angle"] < data["Amplitude"]]
    data = data[data["Mean_angle"] > - data["Amplitude"]]
    return data

if __name__=="__main__":
    fig_folder = "/Users/sintes/Library/CloudStorage/OneDrive-Personal/These/Wobbling/Figures/"
    file = "/Users/sintes/Library/CloudStorage/OneDrive-Personal/These/Wobbling/all_wobbling_data.csv"
    data = load_data(file)
    data["Count_freq"] = 1 / data["Period"]

    filters = ["", "filtered"]

    for filter in filters:
        if filter == "filtered":
            dataLC = filtering_data(data)
        else:
            dataLC = data
        plots(dataLC, filter)


    # plt.show(block=True)
