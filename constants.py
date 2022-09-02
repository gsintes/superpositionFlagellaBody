"""Define the useful constants."""

import os

FOLDER_UP = "Z:/PhD/Ficoll/Ficoll17%_20-11-05_1uLbactos_TRACKING" 
FOLDER_NUM = "2020-11-05_13h14m21s"

FIG_FOLDER = os.path.join(FOLDER_UP, "Wobbling", FOLDER_NUM)
FOLDER = os.path.join(FOLDER_UP, FOLDER_NUM)

IM_SIZE = (1024, 1024)
FPS = 80

MIRE_INFO_PATH = "Z:/PhD/Ficoll/2020-11-05_13h43m12s_mire/mire_info.json"
MIRE_PATH = "Z:/Martyna/PhD/Ficoll/2020-11-05_13h43m12s_mire/Image0574023.tif"


ANGLE_DATA_FILE = "angle_body_flagella.csv"

TRACK_FILE = "Track/Track.txt"