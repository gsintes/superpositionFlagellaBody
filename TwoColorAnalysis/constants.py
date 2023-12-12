"""Define the useful constants."""

import os 

FOLDER_UP = "/Volumes/Chains/Tracking_23-12-08"
# FOLDER_NUM = "2023-09-01_18h54m34s"

# FOLDER_UP = "/Users/sintes/Desktop/Test"
FOLDER_NUM = "2023-07-25_18h14m59s"

EXP_INFO_FILE = os.path.join(FOLDER_UP, "exp-info.csv")

FIG_FOLDER = os.path.join(FOLDER_UP, "Wobbling", FOLDER_NUM)
FOLDER = os.path.join(FOLDER_UP, FOLDER_NUM)

IM_SIZE = (1024, 1024)

MIRE_INFO_PATH = os.path.join(FOLDER_UP, "2023-12-08_17h51m01s_calib/mire_info.json")
MIRE_PATH = f"{FOLDER_UP}/2023-12-08_17h51m01s_calib/Image0018174.tif"


ANGLE_DATA_FILE = "angle_body_flagella.csv"

TRACK_FILE = "Track/Track.txt"