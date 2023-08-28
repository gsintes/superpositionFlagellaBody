"""Test for the superposition."""


import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import constants
from mire_analysis import MireInfo
import superpositionTools as st

class TestSuperposition:
    def setup_method(self):
        self.image = 255 * np.ones(constants.IM_SIZE)
        self.mire_info = MireInfo()
        self.mire_info.middle_line = 530
        self.mire_info.displacement = (2, 15)
        self.track_data = pd.DataFrame({
            "time": [10],
            "x": [1],
            "y": [2],
            "z": [3],
            "center_x": [770],
            "center_y": [520]
        })
        self.position = (self.track_data["center_x"][0], self.track_data["center_y"][0])
        self.image[self.position[0] - 5 : self.position[0] + 5, self.position[1] - 5 : self.position[1] + 5] = 100
        self.red_im, self.green_im = st.split_image(self.image, self.mire_info.middle_line)
        self.center_green = (self.position[0] - self.mire_info.middle_line, self.position[1])
        # self.center_red = (self.position[0] + mire_info)

    def test_image_split(self):
        assert self.red_im.shape == self.green_im.shape

    def test_select_center(self):
        # red_center = st.select_center_image(self.red_im, center=self.center_red, size=100)
        green_center = st.select_center_image(self.green_im, center=self.center_green, size=100)
        # assert red_center.shape == (200, 200)
        assert green_center.shape == (200, 200) 
        check_im = 255 * np.ones((200, 200))
        check_im[95:105, 95:105] = 100

        assert np.array_equal(green_center, check_im)


    def test_superimpose(self):


        plt.figure()
        plt.imshow(self.image)
        plt.title("Image")

        plt.figure()
        plt.imshow(self.red_im)
        plt.title("Red image")

        plt.figure()
        plt.imshow(self.green_im)
        plt.title("Green image")

        plt.show(block=True)