"""Revert by  90deg right rotation all the images."""

import multiprocessing as mp
import argparse
import os

from PIL import Image

def parse_arguments()-> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--recursive", help="""The folder is a folder of folder""", action="store_true")
    parser.add_argument("-f", "--folder", help="""Folder where to revert images?""")
    parser.add_argument("-v", "--verbose", help="Print the progression", action="store_true")
    return parser.parse_args()

def revert_folder(folder: str, verbose: float=False) -> None:
    """Revert the images in the folder."""
    list_im = [os.path.join(folder, f) for f in os.listdir(folder) if (f.endswith(".tif") and not f.startswith("."))]
    for k, im_name in enumerate(list_im):
        if verbose and k % 10 == 0: 
            print(f"{100 * k/len(list_im):.2f}%")
        im = Image.open(im_name)
        out = im.rotate(-90, expand=True)
        out.save(im_name)
    print(f"{folder} done.")

def main(args: argparse.ArgumentParser)-> None:
    folder: str = args.folder
    verbose: bool = args.verbose

    if args.recursive:
        pool = mp.Pool(mp.cpu_count() - 1)
        subfolders = [(os.path.join(folder, sf), False) for sf in os.listdir(folder) if os.path.isdir(os.path.join(folder, sf))]
        for f in subfolders:
            revert_folder(f)
    else:
        revert_folder(folder, verbose)
    

if __name__=="__main__":
    args = parse_arguments()
    main(args)
