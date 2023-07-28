"""Revert by  90deg right rotation all the images in a folder."""
import argparse
import os

from PIL import Image

def parse_arguments()-> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", help="""Folder where to revert images?""")
    parser.add_argument("-v", "--verbose", help="Print the progression", action="store_true")
    return parser.parse_args()

args = parse_arguments()
folder = args.folder

list_im = [os.path.join(folder, f) for f in os.listdir(folder) if (f.endswith(".tif") and not f.startswith("."))]

for k, im_name in enumerate(list_im):
    if args.verbose and k % 10 == 0: 
        print(f"{100 * k/len(list_im):.2f}%")
    im = Image.open(im_name)
    out = im.rotate(-90, expand=True)
    out.save(im_name)
