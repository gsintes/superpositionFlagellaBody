"""Mesure the angle between the body and the flagella along time."""

import os
from typing import Tuple, List

import matplotlib.image as mpim
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology, measure
from skimage.filters import gaussian
from skimage.filters.thresholding import threshold_li
from matplotlib.patches import Ellipse

import superimpose
import constants
from ellipse import LsqEllipse

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
   
    bin_green = li_binarization(blur)
    x, y = keep_bigger_particle(bin_green)
    bin_green = make_bin_im(x, y, bin_green.shape)
    contour = measure.find_contours(bin_green, 0.4)[0]

    X = np.array(list(zip(contour[:, 0], contour[:, 1])))

    reg = LsqEllipse().fit(X)
    center, width, height, phi = reg.as_parameters()
    
    a, b = find_main_axis(x, y)

    a1 = np.tan(phi + np.pi / 2)
    b1 = center[1] - a1 * center[0]

    if visualization:
        ellipse = Ellipse(
            xy=(center[1], center[0]), width=2*width, height=2*height, angle=np.rad2deg(phi - np.pi / 4),
            edgecolor='b', fc='None', lw=2, label='Fit', zorder=2)
        ellipse1 = Ellipse(
            xy=(center[1], center[0]), width=2*width, height=2*height, angle=np.rad2deg(phi - np.pi / 4),
            edgecolor='b', fc='None', lw=2, label='Fit', zorder=2)

        _, axis =plt.subplots(nrows=1, ncols=3)
        plt.suptitle("Body detection")
        x = np.linspace(0, green_im.shape[0])
        axis[0].imshow(green_im, cmap="gray")
        axis[0].set_ylim([green_im.shape[0], 0])
        axis[0].set_xlim([0, green_im.shape[1]])
        axis[0].plot(a * x + b, x, "-g", linewidth=1)  
        axis[0].plot(a1 * x + b1, x, "-b", linewidth=1)  
        axis[0].add_patch(ellipse)
        axis[1].imshow(bin_green, cmap="gray")
        axis[1].add_patch(ellipse1)
        axis[1].set_ylim([green_im.shape[0], 0])
        axis[1].set_xlim([0, green_im.shape[1]])
        axis[1].plot(a * x + b, x, "-g", linewidth=1)        
        axis[1].plot(a1 * x + b1, x, "-b", linewidth=1)  
    return a, b
    
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
    return a1, b1

def detect_angle(
    super_imposed: np.ndarray,
    visualization: bool = False) ->  float:
    """Detect the angle between the body and the flagella."""
    a0, b0 = detect_body(super_imposed[:, :, 1], visualization=True)
    a1, b1 = detect_flagella(super_imposed[:, :, 0], visualization=False)
    x = np.linspace(0, super_imposed.shape[0])
    if visualization:
        
        plt.imshow(super_imposed)
        plt.plot(a0 * x + b0, x, "-g", linewidth=3)
        plt.plot(a1 * x + b1, x, "-r", linewidth=3)
        plt.ylim([super_imposed.shape[0], 0])
        plt.xlim([0, super_imposed.shape[1]])
        plt.draw()
        plt.pause(0.01)
        plt.clf()
        plt.close()
    return np.arctan(a1) - np.arctan(a0)

def save_data(time: List[int], angle: List[float]) -> None:
    """Save the data to a text file."""
    textfile = open(os.path.join(constants.FOLDER, "angle_body_flagella.csv"), "w")
    for i in range(len(time)):
        textfile.write(f"{time[i]}, {angle[i]}\n")
    textfile.close()

def list_angle_detection(
    image_list: List[str],
    visualization: bool = False) -> Tuple[List[float], List[float]]:
    """Run the angle detection on a list of path and return angle and time."""
    angle = []
    time = []
    for i, im_path in enumerate(image_list):
        im_test = mpim.imread(im_path) 
        im_test = im_test / np.amax(im_test)
        super_imposed = superimpose.superposition(im_test, mire_info)
        super_imposed = superimpose.select_center_image(super_imposed, 100)
        time.append(i / constants.FPS)
        angle.append(180 * detect_angle(super_imposed, visualization) / np.pi)
    return time, angle


if __name__ == "__main__":
    mire_info = superimpose.MireInfo(constants.MIRE_INFO_PATH)
    image_list = [os.path.join(constants.FOLDER, f) for f in os.listdir(constants.FOLDER) if (f.endswith(".tif") and not f.startswith("."))]

    im_test = mpim.imread(image_list[1508])
    im_test = superimpose.select_center_image(superimpose.superposition(im_test, mire_info), 100)

    
    green_im = im_test[:, :, 1]
    
    # detect_body(green_im, True)

    # X = np.arange(0, 200, 1)
    # Y = np.arange(0, 200, 1)
    # X, Y = np.meshgrid(X, Y)
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}) 
    # ax.plot_surface(X, Y, green_im)
    # plt.title("Raw")

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}) 
    # ax.plot_surface(X, Y, gaussian(green_im, 2))
    # plt.title("Filtered")

    # detect_angle(im_test, visualization=True)

    time, angle = list_angle_detection(image_list, visualization=True)    
    # save_data(time, angle)
    # plt.close('all')
    # plt.figure()
    # plt.plot(time, angle, ".")
    # plt.xlabel("Time (in s)")
    # plt.ylabel("Angle (in deg)")
    plt.show(block=True)
