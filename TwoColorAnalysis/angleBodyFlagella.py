"""Mesure the angle between the body and the flagella along time."""

import multiprocessing as mp
import os
from typing import Tuple, List
import math

import matplotlib.image as mpim
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, exposure, morphology
from scipy.ndimage import median_filter
import cv2

from skimage.filters import gaussian
from makeTestIm import Rectangle

import superimpose
import utils
from trackParsing import load_track_data, load_info_exp
import body_detection as bd
from mire_info import MireInfo


class NoCenteredParticle(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class Info:
    def __init__(self, folder: str, fps: int, mire_info: MireInfo) -> None:
        self.folder = folder
        self.track_data = load_track_data(self.folder, fps=fps)
        self.mire_info = mire_info
        self.fps = fps
        self.shift = (mire_info.displacement[0] - (1024 // 2),
         - mire_info.middle_line - (1024 - mire_info.middle_line) // 2)


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
    def __init__(self, image: np.ndarray, streched: np.ndarray, bin: np.ndarray, a: float, b: float, rectangle: Rectangle=None) -> None:
        self.image = image
        self.streched = streched
        self.a = a
        self.b = b
        self.bin = bin
        self.rectangle = rectangle

    def __call__(self) -> plt.Figure:
        _, axis =plt.subplots(1, 3)
        x = np.linspace(0, self.image.shape[0])
        axis[0].set_ylim([self.image.shape[0], 0])
        axis[0].set_xlim([0, self.image.shape[1]])
        axis[0].imshow(self.image, cmap="gray")
        axis[0].plot(self.a * x + self.b, x, "-b", linewidth=1)
        axis[0].set_xticks([])
        axis[0].set_yticks([])
        axis[0].set_title("Original")

        axis[1].imshow(self.streched, cmap="gray")
        axis[1].plot(self.a * x + self.b, x, "-b", linewidth=1)
        axis[1].set_ylim([self.image.shape[0], 0])
        axis[1].set_xlim([0, self.image.shape[1]])
        axis[1].set_xticks([])
        axis[1].set_yticks([])
        axis[1].set_title("Contrast enhanced")

        axis[2].imshow(self.bin, cmap="gray")
        axis[2].plot(self.a * x + self.b, x, "-b", linewidth=1)
        axis[2].set_ylim([self.image.shape[0], 0])
        axis[2].set_xlim([0, self.image.shape[1]])
        axis[2].set_xticks([])
        axis[2].set_yticks([])
        axis[2].set_title("Binarized")



        if self.rectangle is not None:
            X, Y = self.rectangle.border((200, 200))
            axis.plot(Y, X, ".r", markersize=2)
            # axis[1].plot(Y, X, ".r", markersize=2)


class AngleDetector:
    """Does the detection of the angle."""
    def __init__(self, fig_folder: str, super_imposed: np.ndarray, i: int, info: Info, visualization: bool) -> None:
        self.fig_folder = fig_folder
        self.super_imposed = super_imposed
        self.i = i
        self.visualization = visualization
        self.center = (info.track_data["center_x"][i], info.track_data["center_y"][i])
        self.vel = (info.track_data["vel_x"][i], info.track_data["vel_y"][i])
        self.green_im = self.super_imposed[:, :, 1]
        self.red_im = self.super_imposed[:, :, 0]

    def detect_flagella(self, visualization: bool = False) -> Tuple[float, float, Tuple[float, float]]:
        """Detect the flagella in the red image."""
        stretched = superimpose.contrast_enhancement(self.red_im)
        filtered = median_filter(stretched, size=15)
        blur = gaussian(filtered, 2)


        cv2.imwrite(f"/Users/sintes/Desktop/Flagella/{self.i}.png", (blur * 0.99 * 2 ** 8 / np.amax(blur)).astype("uint8"))

        bin_red = utils.li_binarization(blur)
        bin_red = morphology.binary_closing(bin_red, footprint=np.ones((9, 9)))
        x, y, _ = keep_bigger_particle(bin_red, center=False)
        bin_red = make_bin_im(x, y, bin_red.shape)

        a, b, vect = find_main_axis(x, y)
        if visualization:
            checker = DetectionChecker(self.red_im, stretched, bin_red, a, b)
            checker()
        return a, b, vect

    def detect_body(self, visualization: bool = False) -> Tuple[float, float]:
        """Detect the body in the green image."""
        detector = bd.BodyDetection(self.green_im, a=40, b=7)
        rectangle = detector(False)

        bin_green = rectangle.make_im((200, 200))
        x, y, _ = keep_bigger_particle(bin_green, center=False)
        bin_green = make_bin_im(x, y, bin_green.shape)

        a, b, vect = find_main_axis(x, y)
        if visualization:
            checker = DetectionChecker(self.green_im, bin_green, a, b, rectangle=rectangle)
            checker()
        return a, b, vect, rectangle

    def __call__(self) -> Tuple[float, float, float, float, float, float]:
        """Detect the angle between the body and the flagella, and the body and velocity."""
        try:
            a0, b0, vect_body, rect = self.detect_body(visualization=False)
            a1, b1, vect_flagella = self.detect_flagella(visualization=False)
        except NoCenteredParticle:
            raise


        # ps_fb = vect_body[0] * vect_flagella[0] + vect_body[1] * vect_flagella[1]
        # sin_theta_fb = - vect_body[0] * vect_flagella[1] + vect_body[1] * vect_flagella[0]
        # angle_fb = np.sign(sin_theta_fb) * np.arccos(ps_fb)
        # if not(math.isnan(self.vel[0]) or math.isnan(self.vel[1])):
        #     vel_norm =  np.sqrt(self.vel[0] ** 2 + self.vel[1] ** 2)
        #     if vel_norm != 0:
        #         ps_vb = (vect_body[0] * self.vel[0] + vect_body[1] * self.vel[1]) / vel_norm
        #         sin_theta_vb = (- vect_body[0] * self.vel[1] + vect_body[1] * self.vel[0]) / vel_norm
        #         angle_vb = np.sign(sin_theta_vb) * np.arccos(ps_vb)
        #     else:
        #         angle_vb = 0
        # else:
        #     angle_vb = 0

        if self.visualization:
            x = np.linspace(0, self.super_imposed.shape[0])
            plt.figure()
            super_imposed_en = superimpose.contrast_enhancement(self.super_imposed)
            # if angle_vb != 0:
            #     plt.quiver(100, 100, self.vel[0], self.vel[1], scale=10)
            plt.imshow(super_imposed_en)
            # plt.plot(a0 * x + b0, x, "-g", linewidth=1)
            plt.plot(a1 * x + b1, x, "-b", linewidth=1)
            X, Y = rect.border((200, 200))
            plt.plot(Y, X, ".r", markersize=2)

            plt.ylim([self.super_imposed.shape[0], 0])
            plt.xlim([0, self.super_imposed.shape[1]])
            # plt.draw()
            # plt.pause(0.001)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(self.fig_folder, f"{self.i}.png"))
        #     # plt.clf()
            plt.close("all")
        return (vect_body[0], vect_body[1], vect_flagella[0], vect_flagella[1], self.vel[0], self.vel[1])


def save_data(angles: List[Tuple[float, float, float, float, float, float, float]], folder) -> None:
    """Save the data to a text file."""
    textfile = open(os.path.join(folder, "angle_body_flagella.csv"), "w")
    textfile.write("time, bodyAxis_x, bodyAxis_y, flagellaAxis_x, flagella_Axis_y, vel_x, vel_y\n")
    for i in range(len(angles)):
        textfile.write(f"{angles[i][0]}, {angles[i][1]}, {angles[i][2]}, {angles[i][3]}, {angles[i][4]}, {angles[i][5]}, {angles[i][6]}\n")
    textfile.close()

def get_center_body(super_imposed: np.ndarray, center_track: Tuple[int, int]):
    """Get the center of the body."""
    green_im = super_imposed[:, :, 1]
    # center_im = (green_im.shape[0] // 2, green_im.shape[1] // 2)
    green_im_centered = superimpose.select_center_image(green_im, center_track, size=200)
    bin_im = utils.li_binarization(green_im_centered)
    _, _, centroid = keep_bigger_particle(bin_im, center=False)
    centroid = (center_track[0] + int(centroid[0]) - 200, center_track[1] + int(centroid[1]) - 200)
    # plt.figure()
    # plt.imshow(green_im, cmap="gray")
    # plt.plot(center_track[1], center_track[0], ".r")
    # plt.plot(centroid[1], centroid[0], "*g")
    # plt.show(block=True)
    return centroid

def analyse_image(fig_folder: str, i: int, image_path: str, info: Info, visualization: bool) -> Tuple[float, float, float, float, float, float, float]:
    """Run the analysis on an image."""
    im_test = mpim.imread(image_path)
    im_test = im_test / 2 ** 16
    super_imposed = superimpose.superposition(im_test, info.mire_info)
    center_track = (int(info.track_data["center_x"][i]) - info.mire_info.middle_line, int(info.track_data["center_y"][i]))
    center = get_center_body(super_imposed, center_track)
    # print(center)
    super_imposed = superimpose.select_center_image(
            super_imposed,
            center=center,
            size=100)
    try:
        detect_angle = AngleDetector(fig_folder, super_imposed, i, info, visualization)
        angles = detect_angle()
        return (i / info.fps, *angles)
    except NoCenteredParticle:
        return (0, 0, 0, 0, 0, 0, 0)

def list_angle_detection(folder_up: str, folder: str,
    image_list: List[str], fps: int,
    visualization: bool = False) -> Tuple[List[float], List[float]]:
    """Run the angle detection on a list of path and return angle and time."""
    fig_folder = os.path.join(folder_up, "Wobbling", folder)
    info = Info(os.path.join(folder_up, folder), fps, mire_info)
    pool = mp.Pool(mp.cpu_count() - 1)
    angles = pool.starmap_async(analyse_image, [(fig_folder, i, image_path, info, visualization) for i, image_path in enumerate(image_list[:len(info.track_data)])]).get()
    pool.close()
    return angles


if __name__ == "__main__":

    folder_up = "/home/guillaume/NAS/SwimmingPVP360/SwimmingPVP_23-09-01"
    mire_info_path = f"{folder_up}/2023-09-01_17h21m57s_calib/mire_info.json"
    exp_info_file = os.path.join(folder_up, "exp-info.csv")
    folder_list = [f for f in os.listdir(folder_up) if not f.startswith(".") and not f.endswith("calib") and os.path.isdir(os.path.join(folder_up, f)) ]
    for folder in folder_list:
        print(folder)
        full_folder = os.path.join(folder_up, folder)
        fig_folder = os.path.join(folder_up, "Wobbling", folder)
        exp_info = load_info_exp(exp_info_file, folder)
        fps = int(exp_info["fps"].values[0])
        visualization = True
        if visualization:
            try:
                os.makedirs(fig_folder)
            except FileExistsError:
                pass
        mire_info = MireInfo(mire_info_path)

        end = int(exp_info["final_flagella_frame"].values[0])

        image_list = [os.path.join(full_folder, f) for f in os.listdir(full_folder) if (f.endswith(".tif") and not f.startswith("."))][0:end]

        angles = list_angle_detection(folder_up, folder, image_list, fps=fps,visualization=visualization)
        save_data(angles, full_folder)
