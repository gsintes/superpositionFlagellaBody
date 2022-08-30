"""Useful functions."""

import time


def timeit(func):
    """Timing decorator."""
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print(t2 - t1)
        return res
    return wrapper