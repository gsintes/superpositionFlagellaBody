"""Useful functions."""

import time

import numpy as np
from skimage.filters.thresholding import threshold_li

def timeit(func):
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
