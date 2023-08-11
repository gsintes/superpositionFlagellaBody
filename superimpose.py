"""Super impose images of the body and the flagella."""

import shutil
from typing import Tuple
import json
import os

import matplotlib.image as mpim
import matplotlib.pyplot as plt
import numpy as np

import constants
import superpositionTools as st

class MireInfo:
    def __init__(self, *args) -> None:
        if len(args) == 2:
            self.middle_line = args[0]
            self.displacement = args[1]
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, dict):
                self.middle_line = arg["middle_line"]
                self.displacement = arg["displacement"]
            if isinstance(arg, str):
                with open(arg) as f:
                    data = json.load(f)
                    self.middle_line = data["middle_line"]
                    self.displacement = data["displacement"]

    def delta_x(self) -> int:
        """Return the displacement in x"""
        return self.displacement[0]
    
    def delta_y(self) -> int:
        """Return the displacement in y"""
        return self.displacement[1]
    
    def save(self, file: str) -> None:
        """Save the mire info in a json file."""
        with open(file, "w", encoding="utf-8") as outfile:
            outfile.write("")
            json.dump(self.__dict__, outfile, indent=4)

    def __repr__(self) -> str:
        return f"Mire info:\n middle_line: {self.middle_line}\n Displacement: {self.displacement}"


def shift_image(
    image: np.ndarray,
    displacement: Tuple[int, int]) -> np.ndarray:
    """Shift the image and fill boundaries with black."""
    if len(image.shape) == 3:
        shifted = image.copy()
        for i in range(3):
            shifted[:, :, i] = shift_image(image[:, :, i], displacement)
        return shifted
    elif len(image.shape) == 2:
        delta_x = displacement[0]
        delta_y = displacement[1]
        if delta_x > 0:
            image = image[delta_x:, :]
            image = np.concatenate((image, np.zeros((delta_x, image.shape[1]))), axis=0)
        if delta_x < 0:
            image = image[:delta_x, :]
            image = np.concatenate((np.zeros((-delta_x, image.shape[1])), image), axis=0)

        if delta_y < 0: 
            image = image[:, :delta_y]
            image = np.concatenate((np.zeros((image.shape[0], -delta_y, )), image), axis=1)
        if delta_y > 0:
            image = image[:, delta_y:]
            image = np.concatenate((image, np.zeros((image.shape[0], delta_y))), axis=1)
        return image
    else:
        raise IndexError("Not the good dimension.")


def super_impose_two_im(
    green_im: np.ndarray,
    red_im: np.ndarray,
    displacement: Tuple[int,int]) -> np.ndarray:
    """Super impose the green and red part."""
    shift_red = shift_image(red_im, displacement)
    super_imposed = np.array([shift_red.transpose(),
     green_im.transpose(),
     np.zeros(green_im.shape).transpose()])
    return super_imposed.transpose()


def superposition(image: np.ndarray, mire_info: MireInfo) -> np.ndarray:
    """Superimpose the two colors according to the info of the mire."""
    red_im, green_im = st.split_image(image, mire_info.middle_line)
    return super_impose_two_im(green_im, red_im, mire_info.displacement)

def folder_superposition(
    folder_im: str,
    folder_save: str,
    mire_info: MireInfo):
    """Run the superposition and save all images in a folder."""
    date_video = folder_im.split("/")
    date_video = date_video[len(date_video) - 1]
    save_dir = os.path.join(folder_save, "Videos_tracking", date_video)
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    image_list = [os.path.join(folder_im, f) for f in os.listdir(folder_im) if (f.endswith(".tif") and not f.startswith("."))]
    for i, im_path in enumerate(image_list):
        im_test = mpim.imread(im_path) 
        im_test = im_test / 2 ** 16
        super_imposed = superposition(im_test, mire_info)
        super_imposed = st.select_center_image(super_imposed)
        super_imposed = st.contrast_enhancement(super_imposed)
        plt.figure()
        plt.imshow(super_imposed, cmap="gray")
        # plt.show(block=True)
        mpim.imsave(os.path.join(save_dir, f"{i}.png"), super_imposed)
        plt.close()


if __name__ == "__main__":
    mire_info = MireInfo(constants.MIRE_INFO_PATH)
    # parent_folder = "/Volumes/GUILLAUME/Ficoll Marty/Ficoll17%_20-11-05_1uLbactos_TRACKING"
    # list_dir = [f for f in os.listdir(parent_folder) if not f.startswith(".")]
    # for folder in list_dir[0: 1]:
        # folder_superposition(os.path.join(parent_folder, folder), "/Users/sintes/Desktop", mire_info)
    folder_superposition(constants.FOLDEs, "/Users/sintes/Desktop", mire_info)