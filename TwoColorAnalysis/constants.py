"""Define the useful constants."""

import os

# FOLDER_UP = "/Users/sintes/Desktop/Martyna/PhD/Ficoll/Ficoll17%_20-11-05_1uLbactos_TRACKING" 
# FOLDER_NUM = "2020-11-05_13h24m47s"

FOLDER_UP = "/Users/sintes/Desktop/NASGuillaume/SwimmingPVP360/SwimmingPVP_23-07-28/"
FOLDER_NUM = "2023-07-28_19h26m26s"

EXP_INFO_FILE = os.path.join(FOLDER_UP, "exp-info.csv")

FIG_FOLDER = os.path.join(FOLDER_UP, "Wobbling", FOLDER_NUM)
FOLDER = os.path.join(FOLDER_UP, FOLDER_NUM)

IM_SIZE = (1024, 1024)
FPS = 80

MIRE_INFO_PATH = f"/Users/sintes/Desktop/NASGuillaume/SwimmingPVP360/SwimmingPVP_23-07-28/2023-07-28_18h15m41s_calib/mire_info.json"
MIRE_PATH = f"{FOLDER_UP}/2023-07-25_18h06m22s-calib/Image0011742.tif"


ANGLE_DATA_FILE = "angle_body_flagella.csv"

TRACK_FILE = "Track/Track.txt"