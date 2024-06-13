"""Look at the relationship between velocity and viscosity."""

from typing import Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

import martinez_data as marti

def vel_model(fluidity: Union[float, np.ndarray], T0: float, Omega_max: float, visc_thresh:float) -> float:
    """Model for velocity as a function of fluidity."""
    if isinstance(fluidity, np.ndarray):
        return np.array([vel_model(f, T0, Omega_max, visc_thresh) for f in fluidity])
    a = 1.5
    b = 0.6
    Omega_c = 3 * T0  / (16 * np.pi * a * b ** 2 * visc_thresh)
    alpha = T0 / (Omega_max - Omega_c)
    visc = 1 / fluidity
    v_0 = 3 * alpha * Omega_max / (16 * np.pi * a * b ** 2 + 3 * alpha)
    if visc > visc_thresh:
        return 3 * T0 * fluidity / (16 * np.pi * a * b ** 2) / v_0
    return 3 * alpha * Omega_max * fluidity / ((16 * np.pi * a * b ** 2 + 3 * alpha * fluidity)) /v_0

def get_v_ref(data: pd.DataFrame) -> Dict[str, float]:
    """Get the velocity at reference viscosity for each date."""
    v_ref = {}
    for date in data.date.unique():
        sub_data = data[(data.date == date) & (data.r_viscosity == 1)]
        if sub_data.empty:
            v_ref[date] = data[data.r_viscosity == 1].vel.mean()
        else:
            v_ref[date] = data[(data.date == date) & (data.r_viscosity == 1)].vel.mean()
    return v_ref


if __name__=="__main__":
    # Load the data
    data = pd.read_csv("/Users/sintes/Desktop/velocity_data.csv")
    data["r_viscosity"] = data["viscosity"].round()
    data["fluidity"] = 1 / data["r_viscosity"]
    data["date"] = data.exp.apply(lambda x: x.split("_")[0])
    v_ref = get_v_ref(data)

    data["vel_norm"] = data.vel / np.array([v_ref[date] for date in data.date])
    # data["vel_norm"] = data["vel"] / v_ref

    my_f = data.fluidity.unique()
    my_v = []
    for f in my_f:
        sub_data = data[data.fluidity == f]
        my_v.append(sub_data.vel_norm.mean())

    x_data = marti.f_pvp360
    x_data = np.concatenate((x_data, marti.f_pvp60))
    x_data = np.concatenate((x_data, 1 / marti.vis_breuer))
    x_data = np.concatenate((x_data, my_f))

    y_data = marti.vel_pvp360
    y_data = np.concatenate((y_data, marti.vel_pvp60))
    y_data = np.concatenate((y_data, marti.vel_breuer))
    y_data = np.concatenate((y_data, my_v))

    popt, pcov = curve_fit(vel_model, x_data, y_data, p0=[1, 1000, 5], bounds=([0, 0, 1], [np.inf, np.inf, 30]))
    print(popt)

    plt.figure(figsize=(10, 6))

    plt.plot(1 / marti.f_pvp360, marti.vel_pvp360, "g*", label="Martinez PVP360")
    plt.plot(1 / marti.f_pvp60, marti.vel_pvp60, "m*", label="Martinez PVP60")
    plt.plot(marti.vis_breuer, marti.vel_breuer, "b*", label="Breuer Ficoll")
    sns.pointplot(x="r_viscosity", y="vel_norm", data=data, native_scale=True, linestyles="", errorbar="se", scale=0.9, err_kws={'linewidth': 1.5}, c="k")
    sns.scatterplot(x="r_viscosity", y="vel_norm", data=data, label="PVP 360", s=5, c="k")

    plt.legend()
    plt.xlabel("$\eta_0 / \eta$")
    plt.ylabel("$V / V_0$")

    plt.figure(figsize=(10, 6))
    x = np.linspace(0.01, 1, 100)
    plt.plot(x, vel_model(x, *popt), "r-", label="Model")
    plt.plot(marti.f_pvp360, marti.vel_pvp360, "g*", label="Martinez PVP360")
    plt.plot(marti.f_pvp60, marti.vel_pvp60, "m*", label="Martinez PVP60")
    plt.plot(1 / marti.vis_breuer, marti.vel_breuer, "b*", label="Breuer Ficoll")
    sns.pointplot(x="fluidity", y="vel_norm", data=data, native_scale=True, linestyles="", errorbar="se", scale=0.9, err_kws={'linewidth': 1.5}, c="k")
    sns.scatterplot(x="fluidity", y="vel_norm", data=data, label="PVP 360", s=5, c="k")

    plt.legend()
    plt.xlabel("$\eta / \eta_0$")
    plt.ylabel("$V / V_0$")

    # plt.show(block=True)


