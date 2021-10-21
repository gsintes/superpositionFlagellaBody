"""Mesure the angle between the body and the flagella along time."""

import os

import matplotlib.image as mpim
import matplotlib.pyplot as plt
import numpy as np
from superimpose import binarize

import superimpose


MIRE_PATH = "/Volumes/GUILLAUME/Ficoll Marty/2020-11-05_13h43m12s_mire/Image0574023.tif"
FOLDER = "/Volumes/GUILLAUME/Ficoll Marty/Ficoll17%_20-11-05_1uLbactos_TRACKING/2020-11-05_13h15m43s"
IM_SIZE = (1024, 1024)

if __name__ == "__main__":
    mire_info = superimpose.mire_analysis(MIRE_PATH)

