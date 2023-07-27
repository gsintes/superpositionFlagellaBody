"""Revert all the images in a folder."""

import os

from PIL import Image
folder = "/Volumes/GuillaumeHD/SwimmingPVP_23_07_25/2023-07-25_18h12m38s"
list_im = [os.path.join(folder, f) for f in os.listdir(folder) if (f.endswith(".tif") and not f.startswith("."))]

for k, im_name in enumerate(list_im):
    if k % 10 == 0: 
        print(f"{100 * k/len(list_im):.2f}%")
    im = Image.open(im_name)
    out = im.rotate(-90, expand=True)
    out.save(im_name)
