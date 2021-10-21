"""Mesure the angle between the body and the flagella along time."""

import os

import matplotlib.image as mpim
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters.thresholding import threshold_otsu


import superimpose


MIRE_PATH = "/Volumes/GUILLAUME/Ficoll Marty/2020-11-05_13h43m12s_mire/Image0574023.tif"
IM_SIZE = (1024, 1024)
