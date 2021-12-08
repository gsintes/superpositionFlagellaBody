"""Select the zones where the detection works for all folder."""

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

import constants
import analysis_angle

def select_region(folder: str) -> Tuple[int, int]:
    """Select the good region on the plot."""
    time, angle = analysis_angle.load_data(folder)
    time, angle = analysis_angle.clean_data(time, angle, 0.5)
    plt.figure(figsize=(12, 6))
    plt.plot(time, angle, "-")
    plt.xlabel("Time (in s)")
    plt.ylabel("Angle (in degrees)")
    coord = plt.ginput(2)
    index_0 = round((coord[0][0] - min(time)) * constants.FPS)
    index_1 = round((coord[1][0] - min(time)) * constants.FPS)
    plt.close()
    return index_0, index_1

if __name__ == "__main__":
    list_dir = [os.path.join(constants.FOLDER_UP, f) for f in os.listdir(constants.FOLDER_UP) if not (f.startswith(".") or f.endswith(".csv"))]
    folders: List[str] = []
    limits: List[Tuple[int, int]] = []
    for folder in list_dir:
        again = True
        while again:
            try:
                index_0, index_1 = select_region(folder)
            except IndexError:
                inp = input("Do you want to skip this plot? (Y/n)")
                if not (inp == "Y" or inp == "y"):
                    index_0, index_1 = select_region(folder)
            folders.append(folder)
            limits.append((index_0, index_1))
            inp = input("Do you want to select another region on this plot ? (Y/n)")
            again = (inp == "Y" or inp == "y")
    data = pd.DataFrame()
    data["Folder"] = folders
    data["Limits"] = limits
    data.to_csv(os.path.join(constants.FOLDER_UP, "wobbling_data.csv"))
        