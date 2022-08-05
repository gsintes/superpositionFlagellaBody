"""Mesure the angle between the body and the flagella along time."""

from ast import Try
import os
from typing import Tuple, List

import matplotlib.image as mpim
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology, measure
from skimage.filters import gaussian
from skimage.filters.thresholding import threshold_li

import superimpose
import constants
from trackParsing import load_track_data


class NoCenteredParticle(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def li_binarization(image: np.ndarray) -> np.ndarray:
    """Binarize the image using the li algorithm."""
    t = threshold_li(image)
    return 1 * (image > t)


def pca(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the main component of the detected region."""
    M = np.cov(x, y)
    val, vect = np.linalg.eig(M) 
    val = list(val)
    i = val.index(max(val))
    return vect[:, i]


def keep_bigger_particle(bin_image: np.ndarray, center: bool):
    """Keep only the bigger particle in the image if it is close to the center."""
    labeled = measure.label(bin_image)
    props = measure.regionprops_table(labeled, properties=("area", "coords","centroid"))
    ok = False
    areas = list(props["area"])
    if not areas:
        raise NoCenteredParticle
    ind_max = areas.index(max(areas))
    if center:
        i = 0
        while not ok and i < len(areas) + 1:
            ind_max = areas.index(max(areas))
            centroid = (props["centroid-0"][ind_max], props["centroid-1"][ind_max])
            ok = (120 > centroid[0] > 80) and (120 > centroid[1] > 80)
            if not ok:
                areas[ind_max] = 0
            i += 1
        if i == len(areas) + 1 or areas[ind_max] < 5:
            raise NoCenteredParticle

    centroid = (props["centroid-0"][ind_max], props["centroid-1"][ind_max])
    coords = props["coords"][ind_max]
    x = coords[:, 0]
    y = coords[:, 1]
    return x, y, centroid


def make_bin_im(X: np.ndarray, Y: np.ndarray, shape:Tuple[int, int]) -> np.ndarray:
    """Make a binary image from the coord of the white points."""
    image = np.zeros(shape)
    image[X, Y] = 1
    return image


def find_main_axis(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, Tuple[float, float]]:
    """
    Find the main axis of a set of points using pca.
    
    Returns: a, b: coeff of the line : y = a * x + b
    """
    vect = pca(x, y)
    if vect[0] != 0:
        a = vect[1] / vect[0]
    else: 
        a =0
    b = np.mean(y) - a * np.mean(x)
    return a, b, vect


class DetectionChecker:
    """Does the image for checking axis detection."""
    def __init__(self, image: np.ndarray, bin: np.ndarray, a: float, b: float) -> None:
        self.image = image
        self.a = a
        self.b = b
        self.bin = bin

    def __call__(self) -> plt.Figure:
        _, axis =plt.subplots(1, 3)
        x = np.linspace(0, self.image.shape[0])
        axis[0].set_ylim([self.image.shape[0], 0])
        axis[0].set_xlim([0, self.image.shape[1]])
        axis[0].imshow(self.image, cmap="gray")
        axis[0].plot(self.a * x + self.b, x, "-b", linewidth=1)

        axis[1].imshow(self.bin, cmap="gray")
        axis[1].plot(self.a * x + self.b, x, "-b", linewidth=1)
        axis[1].set_ylim([self.image.shape[0], 0])
        axis[1].set_xlim([0, self.image.shape[1]])


class AngleDetector:
    """Does the detection of the angle."""
    def __init__(self, super_imposed: np.ndarray, i: int, visualization: bool) -> None:
        self.super_imposed = super_imposed
        self.i = i
        self.visualization = visualization

        self.green_im = self.super_imposed[:, :, 1]
        self.red_im = self.super_imposed[:, :, 0]

    def detect_flagella(self, visualization: bool = False) -> Tuple[float, float, Tuple[float, float]]: 
        """Detect the flagella in the red image."""
        blur = gaussian(self.red_im, 2)
        bin_red = li_binarization(blur)
        x, y, _ = keep_bigger_particle(bin_red, center=False)
        bin_red = make_bin_im(x, y, bin_red.shape)

        a, b, vect = find_main_axis(x, y)
        if visualization:
            checker = DetectionChecker(self.green_im, bin_red, a, b) 
            checker()
        return a, b, vect
        
    def detect_body(self, visualization: bool = False) -> Tuple[float, float]:
        """Detect the body in the green image."""
        # footprint = morphology.disk(1)
        # res = morphology.white_tophat(self.green_im, footprint)
        # res = self.green_im - res
        blur = gaussian(self.green_im, 1)
    
        bin_green = li_binarization(blur)
        x, y, _ = keep_bigger_particle(bin_green, center=True)
        bin_green = make_bin_im(x, y, bin_green.shape)

        a, b, vect = find_main_axis(x, y)
        if visualization:
            checker = DetectionChecker(self.green_im, bin_green, a, b) 
            checker()
        return a, b, vect

    def __call__(self) -> float    :
        """Detect the angle between the body and the flagella."""
        try:
            a0, b0, vect_body = self.detect_body(visualization=True)
            a1, b1, vect_flagella = self.detect_flagella(visualization=False)
        except NoCenteredParticle:
            raise
        x = np.linspace(0, self.super_imposed.shape[0])

        ps = vect_body[0] * vect_flagella[0] + vect_body[1] * vect_flagella[1]
        sin_theta = - vect_body[0] * vect_flagella[1] + vect_body[1] * vect_flagella[0]

        if self.visualization:
            super_imposed_en = superimpose.contrast_enhancement(self.super_imposed)
            plt.imshow(super_imposed_en)
            plt.plot(a0 * x + b0, x, "-g", linewidth=1)
            plt.plot(a1 * x + b1, x, "-r", linewidth=1)
            plt.ylim([self.super_imposed.shape[0], 0])
            plt.xlim([0, self.super_imposed.shape[1]])
            # plt.draw()
            plt.pause(0.001)
            plt.savefig(f"/Users/sintes/Desktop/Wobbling1/{self.i}")      
            plt.clf()
            plt.close() 
        return np.sign(sin_theta) * np.arccos(ps)


def save_data(time: List[int], angle: List[float], folder=constants.FOLDER) -> None:
    """Save the data to a text file."""
    textfile = open(os.path.join(folder, "angle_body_flagella.csv"), "w")
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
    angles = []
    times = []
    stored = []
    for i, im_path in enumerate(image_list[:len(track_data)]):
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
        try:
            detect_angle = AngleDetector(super_imposed, i, visualization)
            angles.append(180 * detect_angle() / np.pi)
            times.append(i / constants.FPS)
        except NoCenteredParticle:
            pass
    return times, angles


if __name__ == "__main__":
    mire_info = superimpose.MireInfo(constants.MIRE_INFO_PATH)
    image_list = [os.path.join(constants.FOLDER, f) for f in os.listdir(constants.FOLDER) if (f.endswith(".tif") and not f.startswith("."))]

    times, angle = list_angle_detection(image_list, window_size=20, visualization=True)    
    save_data(times, angle, constants.FOLDER)
