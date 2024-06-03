"""Look at the relationship between velocity and viscosity."""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from trackParsing import load_track_data, load_info_exp
import martinez_data as marti

parent_folder = "/Users/sintes/Desktop/NASGuillaume/SwimmingPVP360"

subfolders = [os.path.join(parent_folder, f)
              for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f)) and f.startswith("SwimmingPVP")]

viscosity_list: List[float] = []
velocity_list: List[float] = []

for subfolder in subfolders:
    experiments = [f for f in os.listdir(subfolder) if f.startswith("202") and os.path.isdir(os.path.join(subfolder, f))
                   and not(f.endswith("calib"))]
    for exp in experiments:
        exp_info = load_info_exp(os.path.join(subfolder, "exp-info.csv"), exp)
        viscosity = exp_info["viscosity"].values[0]
        limit = int(exp_info["final_valid_frame"].values[0])
        fps = exp_info["fps"].values[0]
        viscosity_list.append(viscosity)
        track_data = load_track_data(os.path.join(subfolder, exp), fps=fps)
        track_data = track_data.iloc[:limit]
        vel = track_data["vel"].mean()
        velocity_list.append(vel)

viscosity_list = np.array(viscosity_list)
velocity_list = np.array(velocity_list)

vis_unique = list(set(viscosity_list))
vis_unique.sort()
vis_unique = np.array(vis_unique)
vel_grouped = np.zeros(vis_unique.shape)
std = np.zeros(vis_unique.shape)

for k, vis in enumerate(vis_unique):
    temp = []
    for i, v in enumerate(viscosity_list):
        if v==vis:
            temp.append(velocity_list[i])
    vel_grouped[k] = np.mean(temp)
    std[k] = np.std(temp)


plt.figure()
plt.plot(viscosity_list, velocity_list, "s")

plt.figure()
plt.errorbar(x=1 / vis_unique, y=vel_grouped / vel_grouped[0], yerr=std / vel_grouped[0], marker="s", linestyle="", label="PVP360")
plt.plot(marti.f_pvp360, marti.vel_pvp360, "*", label="Martinez PVP360")
plt.plot(marti.f_pvp60, marti.vel_pvp60, "*", label="Martinez PVP60")
plt.legend()
plt.xlabel("$\eta_0 / \eta$")
plt.ylabel("$V / V_0$")

plt.show(block=True)


