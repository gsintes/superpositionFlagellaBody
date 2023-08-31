"""Define the useful constants."""

import os

# FOLDER_UP = "/Users/sintes/Desktop/Martyna/PhD/Ficoll/Ficoll17%_20-11-05_1uLbactos_TRACKING" 
# FOLDER_NUM = "2020-11-05_13h24m47s"

# FOLDER_UP = "/Users/sintes/Desktop/NASGuillaume/SwimmingPVP360/SwimmingPVP_23-08-01/"
# FOLDER_NUM = "2023-07-25_18h15m56s"

FOLDER_UP = "/Users/sintes/Desktop/Test"
FOLDER_NUM = "2023-07-25_18h14m59s"

EXP_INFO_FILE = os.path.join(FOLDER_UP, "exp-info.csv")

FIG_FOLDER = os.path.join(FOLDER_UP, "Wobbling", FOLDER_NUM)
FOLDER = os.path.join(FOLDER_UP, FOLDER_NUM)

IM_SIZE = (1024, 1024)
FPS = 80

MIRE_INFO_PATH = os.path.join(FOLDER_UP, "2023-07-25_18h06m22s-calib/mire_info.json")
MIRE_PATH = f"{FOLDER_UP}/2023-07-25_18h06m22s-calib/Image0003873.tif"


ANGLE_DATA_FILE = "angle_body_flagella.csv"

TRACK_FILE = "Track/Track.txt"