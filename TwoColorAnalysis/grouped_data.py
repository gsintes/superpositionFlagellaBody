"""Group all the velocity data."""

import os
from typing import List

import pandas as pd

from trackParsing import load_track_data, load_info_exp


if __name__ == "__main__":
    parent_folder = "/Volumes/Guillaume /SwimmingPVP360"
    subfolders = [os.path.join(parent_folder, f)
              for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f)) and f.startswith("SwimmingPVP")]
    data: List[pd.DataFrame] = []
    for subfolder in subfolders:
        experiments = [f for f in os.listdir(subfolder) if f.startswith("202") and os.path.isdir(os.path.join(subfolder, f))
                    and not(f.endswith("calib"))]
        for exp in experiments:
            exp_info = load_info_exp(os.path.join(subfolder, "exp-info.csv"), exp)
            limit = int(exp_info["final_valid_frame"].values[0])
            fps = exp_info["fps"].values[0]
            track_data = load_track_data(os.path.join(subfolder, exp), fps=fps)
            track_data = track_data.iloc[:limit]
            vel = track_data["vel"].mean()
            exp_info["vel"] = vel
            data.append(exp_info)
    data_pd = pd.concat(data)
    data_pd.to_csv(os.path.join(parent_folder, "velocity_data.csv"))