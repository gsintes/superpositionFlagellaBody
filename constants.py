"""Define the useful constants."""

import os

FOLDER_UP = "/Users/sintes/Desktop/NASGuillaume/SwimmingPVP360/SwimmingPVP_23-07-25" 
FOLDER_NUM = "2023-07-25_18h14m59s"

EXP_INFO_FILE = os.path.join(FOLDER_UP, "exp-info.csv")

FIG_FOLDER = os.path.join(FOLDER_UP, "Wobbling", FOLDER_NUM)
FOLDER = os.path.join(FOLDER_UP, FOLDER_NUM)

IM_SIZE = (1024, 1024)
FPS = 80

MIRE_INFO_PATH = f"{FOLDER_UP}/2023-07-25_18h06m22s-calib/mire_info.json"
MIRE_PATH = f"{FOLDER_UP}/2023-07-25_18h06m22s-calib/Image0011742.tif"


ANGLE_DATA_FILE = "angle_body_flagella.csv"

TRACK_FILE = "Track/Track.txt"