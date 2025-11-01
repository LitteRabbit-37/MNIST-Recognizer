import numpy as np


def normalize_pixels(pixels):
    """Normalize a pixel array [0,255] to [0,1]"""
    return np.array(pixels) / 255.0


# Other helpers can be added here
