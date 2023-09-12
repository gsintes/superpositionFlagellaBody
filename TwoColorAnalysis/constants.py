"""Define the useful constants."""

import os

FOLDER_UP = "/Users/sintes/Desktop/NASGuillaume/SwimmingPVP360/SwimmingPVP_23-08-01/"
FOLDER_NUM = "2023-09-01_17h59m58s"

# FOLDER_UP = "/Users/sintes/Desktop/Test"
# FOLDER_NUM = "2023-07-25_18h14m59s"

EXP_INFO_FILE = os.path.join(FOLDER_UP, "exp-info.csv")

FIG_FOLDER = os.path.join(FOLDER_UP, "Wobbling", FOLDER_NUM)
FOLDER = os.path.join(FOLDER_UP, FOLDER_NUM)

IM_SIZE = (1024, 1024)
FPS = 80

MIRE_INFO_PATH = os.path.join(FOLDER_UP, "2023-08-01_17h23m47s_calib/mire_info.json")
MIRE_PATH = f"{FOLDER_UP}/2023-09-01_17h21m57s_calib/Image0011396.tif"


ANGLE_DATA_FILE = "angle_body_flagella.csv"

TRACK_FILE = "Track/Track.txt"