"""Select the zones where the detection works for all folder."""

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

import constants
class EmptyDataException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def select_region(folder: str) -> Tuple[int, int]:
    """Select the good region on the plot."""
    data = pd.read_csv(os.path.join(folder, "angle_body_flagella.csv"))
    time = data["Time"]
    angle = data[" FlagellaBody angle"]
    # time, angle = analysis_angle.clean_data(time, angle, 0.5)
    if len(time) > 0:
        plt.figure(figsize=(12, 6))
        plt.title(folder.split("/")[-1])
        plt.plot(time, angle, "-")
        plt.xlabel("Time (in s)")
        plt.ylabel("Angle (in degrees)")
        coord = plt.ginput(2)
        index_0 = round((coord[0][0] - min(time)) * constants.FPS)
        index_1 = round((coord[1][0] - min(time)) * constants.FPS)
        plt.close()
        return index_0, index_1
    raise EmptyDataException

if __name__ == "__main__":
    list_dir = [os.path.join(constants.FOLDER_UP, f) for f in os.listdir(constants.FOLDER_UP) if not (f.startswith(".") or f.endswith(".csv"))]
    folders: List[str] = []
    limits: List[Tuple[int, int]] = []
    for folder in list_dir:
        again = True
        while again:
            try:
                index_a, index_b = select_region(folder)
                index_0 = min(index_a, index_b)
                index_1 = max(index_a, index_b)
                if index_1 != index_0:
                    folders.append(folder)
                    if index_0 > 0:
                        limits.append((index_0, index_1))
                    else:
                        limits.append((0, index_1))
                inp = input("Do you want to select another region on this plot ? (Y/n)")
                again = (inp == "Y" or inp == "y")
            except IndexError:

                inp = input("Do you want to skip this plot? (Y/n)")
                if (inp == "Y" or inp == "y"):
                    again = False
            except FileNotFoundError:
                print(f"Data file not found for {folder}")
                again = False
            except EmptyDataException:
                again = False

    data = pd.DataFrame()
    data["Folder"] = folders
    data["Limits"] = limits
    data.to_csv(os.path.join(constants.FOLDER_UP, "wobbling_data.csv"), index=False)
