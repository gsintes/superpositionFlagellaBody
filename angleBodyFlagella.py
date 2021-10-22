"""Mesure the angle between the body and the flagella along time."""

import os
from typing import Tuple

import matplotlib.image as mpim
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian

import superimpose


MIRE_INFO_PATH = "/Volumes/GUILLAUME/Ficoll Marty/2020-11-05_13h43m12s_mire/mire_info.json"
FOLDER = "/Volumes/GUILLAUME/Ficoll Marty/Ficoll17%_20-11-05_1uLbactos_TRACKING/2020-11-05_13h15m43s"
IM_SIZE = (1024, 1024)

def detect_body(
    green_im: np.ndarray,
    visualization: bool = False) -> Tuple[float, float]:
    """Detect the body in the green image."""
    blur = gaussian(green_im, 2)
    bin_green = superimpose.binarize(blur)
    x = []
    y = []
    for i in range(bin_green.shape[0]):
        for j in range(bin_green.shape[1]):
            if bin_green[i, j] == 1:
                x.append(i)
                y.append(j)
    x = np.array(x)
    y = np.array(y)
    a, b =np.polyfit(x, y, 1)
    if visualization:
        _, axis =plt.subplots(1, 2)
        axis[0].imshow(green_im, cmap="gray")
        axis[0].plot(a * x + b, x, "-g", linewidth=3)
        axis[1].imshow(bin_green, cmap="gray")
        axis[1].plot(a * x + b, x, "-g", linewidth=3)
    return (a, b)





if __name__ == "__main__":
    mire_info = superimpose.MireInfo(MIRE_INFO_PATH)
    image_list = [os.path.join(FOLDER, f) for f in os.listdir(FOLDER) if (f.endswith(".tif") and not f.startswith("."))]
    sub_list_images = image_list[1507: 1572]

    im_test = mpim.imread(sub_list_images[40]) 
    im_test = im_test / np.amax(im_test)
    super_imposed = superimpose.superposition(im_test, mire_info)
    super_imposed = superimpose.select_center_image(super_imposed, 100)

    green_im = super_imposed[:,:, 1]
    mpim.imsave("green_im.png", green_im, cmap="gray")
    detect_body(green_im, visualization=True)

    plt.show(block=True)
