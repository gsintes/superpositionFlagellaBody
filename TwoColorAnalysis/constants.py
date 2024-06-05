"""Define the useful constants."""

import os

FOLDER_UP = "/Volumes/Chains/GuillaumeJanvier"
# FOLDER_NUM = "2023-09-01_18h54m34s"

# FOLDER_UP = "/Users/sintes/Desktop/Test"
FOLDER_NUM = "2023-07-25_18h14m59s"

EXP_INFO_FILE = os.path.join(FOLDER_UP, "exp-info.csv")

FIG_FOLDER = os.path.join(FOLDER_UP, "Wobbling", FOLDER_NUM)
FOLDER = os.path.join(FOLDER_UP, FOLDER_NUM)

IM_SIZE = (1024, 1024)

MIRE_PATH = f"{FOLDER_UP}/2024-01-05_16h40m04s_calib/Image0002031.tif"


ANGLE_DATA_FILE = "angle_body_flagella.csv"

TRACK_FILE = "Track/Track.txt"