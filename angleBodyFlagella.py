"""Mesure the angle between the body and the flagella along time."""

import os
from typing import Tuple, List
import time

import matplotlib.image as mpim
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology, measure
from skimage.filters import gaussian
from skimage.filters.thresholding import threshold_li

import superimpose
import constants
from trackParsing import load_track_data

def timit(func):
    """Timing decorator."""
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print(t2 - t1)
        return res
    return wrapper

def li_binarization(image: np.ndarray) -> np.ndarray:
    """Binarize the image using the li algorithm."""
    t = threshold_li(image)
    return 1 * (image > t)


def pca(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return the main component of the detected region."""
    M = np.cov(X, Y)
    val, vect = np.linalg.eig(M)
    val = list(val)
    i = val.index(max(val))
    return vect[:, i]


def keep_bigger_particle(bin_image: np.ndarray):
    """Keep only the bigger particle in the image."""
    labeled = measure.label(bin_image)
    props = measure.regionprops_table(labeled, properties=("area", "coords"))
    coords = props["coords"][list(props["area"]).index(max(props["area"]))]
    
    x = coords[:, 0]
    y = coords[:, 1]
    return x, y


def make_bin_im(X: np.ndarray, Y: np.ndarray, shape:Tuple[int, int]) -> np.ndarray:
    """Make a binary image from the coord of the white points."""
    image = np.zeros(shape)
    image[X, Y] = 1
    return image


def find_main_axis(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Find the main axis of a set of points using pca.
    
    Returns: a, b: coeff of the line : y = a * x + b
    """
    vect = pca(x, y)
    a = vect[1] / vect[0]
    b = np.mean(y) - a * np.mean(x)
    return a, b


def detect_body(
    green_im: np.ndarray,
    visualization: bool = False) -> Tuple[float, float]:
    """Detect the body in the green image."""
    footprint = morphology.disk(1)
    res = morphology.white_tophat(green_im, footprint)
    res = green_im - res
    blur = gaussian(res, 3)
    # blur = res

   
    bin_green = li_binarization(blur)
    x, y = keep_bigger_particle(bin_green)
    bin_green = make_bin_im(x, y, bin_green.shape)
    contour = measure.find_contours(bin_green, 0.4)[0]

    X = np.array(list(zip(contour[:, 0], contour[:, 1])))

    vect = pca(x, y)
    a = vect[1] / vect[0]
    b = np.mean(y) - a * np.mean(x)
    if visualization:
        _, axis =plt.subplots(nrows=1, ncols=3)
        plt.suptitle("Body detection")
        x = np.linspace(0, green_im.shape[0])
        axis[0].imshow(green_im, cmap="gray")
        axis[0].set_ylim([green_im.shape[0], 0])
        axis[0].set_xlim([0, green_im.shape[1]])
        axis[0].plot(a * x + b, x, "-g", linewidth=1)  
        axis[1].imshow(bin_green, cmap="gray")
        axis[1].set_ylim([green_im.shape[0], 0])
        axis[1].set_xlim([0, green_im.shape[1]])
        axis[1].plot(a * x + b, x, "-g", linewidth=1)        
    return a, b, vect

    
def detect_flagella(
    red_im: np.ndarray,
    visualization: bool = False) -> Tuple[float, float]:
    """Detect the flagella in the red image."""
    blur = gaussian(red_im, 2)
    bin_red = li_binarization(blur)
    x, y = keep_bigger_particle(bin_red)
    bin_red = make_bin_im(x, y, bin_red.shape)

    vect = pca(x, y)
    a1 = vect[1] / vect[0]
    b1 = np.mean(y) - a1 * np.mean(x)
    if visualization:
        _, axis =plt.subplots(1, 3)
        plt.suptitle("Flagella detection")
        axis[0].set_ylim([red_im.shape[0], 0])
        axis[0].set_xlim([0, red_im.shape[1]])
        axis[0].imshow(red_im, cmap="gray")
        axis[0].plot(a1 * x + b1, x, "-r", linewidth=1)
        axis[1].imshow(bin_red, cmap="gray")
        axis[1].plot(a1 * x + b1, x, "-r", linewidth=1)
        axis[1].set_ylim([red_im.shape[0], 0])
        axis[1].set_xlim([0, red_im.shape[1]])
    return a1, b1, vect


def detect_angle(
    super_imposed: np.ndarray,
    visualization: bool = False) ->  float:
    """Detect the angle between the body and the flagella."""
    a0, b0, vect_body = detect_body(super_imposed[:, :, 1], visualization=visualization)
    a1, b1, vect_flagella = detect_flagella(super_imposed[:, :, 0], visualization=False)
    x = np.linspace(0, super_imposed.shape[0])
    ps = vect_body[0] * vect_flagella[0] + vect_body[1] * vect_flagella[1]
    if visualization:
        
        plt.imshow(super_imposed)
        plt.plot(a0 * x + b0, x, "-g", linewidth=3)
        plt.plot(a1 * x + b1, x, "-r", linewidth=3)
        plt.ylim([super_imposed.shape[0], 0])
        plt.xlim([0, super_imposed.shape[1]])
        plt.draw()
        plt.pause(0.001)
        plt.clf()
        plt.close() 
    return np.arccos(ps)


def save_data(time: List[int], angle: List[float]) -> None:
    """Save the data to a text file."""
    textfile = open(os.path.join(constants.FOLDER, "angle_body_flagella.csv"), "w")
    for i in range(len(time)):
        textfile.write(f"{time[i]}, {angle[i]}\n")
    textfile.close()


def list_angle_detection(
    image_list: List[str],
    window_size: int,
    visualization: bool = False) -> Tuple[List[float], List[float]]:
    """Run the angle detection on a list of path and return angle and time."""
    track_data = load_track_data()
    shift_y = - mire_info.middle_line - (constants.IM_SIZE[1] - mire_info.middle_line) // 2
    shift_x =  mire_info.displacement[0] - (constants.IM_SIZE[1] // 2)
    angle = []
    times = []
    stored = []
    for i, im_path in enumerate(image_list):
        t1 = time.time()
        im_test = mpim.imread(im_path) 
        im_test = im_test / 2 ** 16
        delta_x = int(track_data["center_x"][i]) + shift_x
        delta_y = int(track_data["center_y"][i]) + shift_y 
        super_imposed = superimpose.shift_image(superimpose.superposition(im_test, mire_info),(-delta_x, -delta_y))
        super_imposed = superimpose.select_center_image(super_imposed, 100) 
        
        stored.append(super_imposed.copy())
        if i >= window_size:
            stored.pop(0)
        for k in range(super_imposed.shape[0]):
            for l in range(super_imposed.shape[1]):
                super_imposed[k, l, 1] = np.mean([image[k,l, 1] for image in stored])
        t3 = time.time()

        times.append(i / constants.FPS)
        angle.append(180 * detect_angle(super_imposed, visualization) / np.pi)
        t2 = time.time()
        print(t2 - t1)
    return times, angle


if __name__ == "__main__":
    mire_info = superimpose.MireInfo(constants.MIRE_INFO_PATH)
    image_list = [os.path.join(constants.FOLDER, f) for f in os.listdir(constants.FOLDER) if (f.endswith(".tif") and not f.startswith("."))]

    times, angle = list_angle_detection(image_list, window_size=20, visualization=True)    
    save_data(times, angle)
    plt.close('all')
    plt.show(block=True)
