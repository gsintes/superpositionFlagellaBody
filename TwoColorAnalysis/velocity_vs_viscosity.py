"""Look at the relationship between velocity and viscosity."""

from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from trackParsing import load_track_data, load_info_exp
import martinez_data as marti


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

# Load the data
data = pd.read_csv("/Users/sintes/Desktop/velocity_data.csv")
data["r_viscosity"] = data["viscosity"].round()
data["date"] = data.exp.apply(lambda x: x.split("_")[0])
v_ref = get_v_ref(data)

data["vel_norm"] = data.vel / np.array([v_ref[date] for date in data.date])
# data["vel_norm"] = data["vel"] / v_ref


plt.figure(figsize=(10, 6))

plt.plot(1 / marti.f_pvp360, marti.vel_pvp360, "g*", label="Martinez PVP360")
plt.plot(1 / marti.f_pvp60, marti.vel_pvp60, "m*", label="Martinez PVP60")
sns.pointplot(x="r_viscosity", y="vel_norm", data=data, native_scale=True, linestyles="", errorbar="se", scale=0.9, err_kws={'linewidth': 1.5}, c="k")
sns.scatterplot(x="r_viscosity", y="vel_norm", data=data, label="PVP 360", s=5, c="k")


plt.legend()
plt.xlabel("$\eta / \eta_0$")
plt.ylabel("$V / V_0$")

plt.show(block=True)


