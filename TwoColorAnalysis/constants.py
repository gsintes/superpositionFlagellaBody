"""Define the useful constants."""

import os

FOLDER_UP = "/home/guillaume/NAS/SwimmingPVP360/SwimmingPVP_23-07-25"

FOLDER_NUM = "2023-07-25_18h18m53s"

EXP_INFO_FILE = os.path.join(FOLDER_UP, "exp-info.csv")

FIG_FOLDER = os.path.join(FOLDER_UP, "Wobbling", FOLDER_NUM)
FOLDER = os.path.join(FOLDER_UP, FOLDER_NUM)

IM_SIZE = (1024, 1024)

# MIRE_PATH = f"{FOLDER_UP}/2024-01-05_16h40m04s_calib/Image0002031.tif"
MIRE_INFO_PATH = f"{FOLDER_UP}/2023-07-25_18h06m22s_calib/mire_info.json"

ANGLE_DATA_FILE = "angle_body_flagella.csv"

TRACK_FILE = "Track/Track.txt"