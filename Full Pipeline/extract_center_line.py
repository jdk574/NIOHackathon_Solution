import cv2
import numpy as np
import matplotlib.pyplot as plt

"""Small custom thinning helper used when cleaning detected axis masks."""

def kernel_func(patch):
    if (patch[1,0]>0 and patch[1,2]>0) and (patch[0,1]==0 and patch[2,1]==0):
        return False
    elif (patch[0,1]>0 and patch[2,1]>0) and (patch[1,0]==0 and patch[1,2]==0):
        return False
    else:
        return True


def apply_custom_kernel_rule(img):
    """
    img: 2D numpy array (binary or grayscale)
    kernel_func: function that takes a local patch and returns True/False

    returns: modified image
    """

    img = img.copy()
    h, w = img.shape

    out = img.copy()

    # assume 3x3 kernel (can generalize later)
    k = 1  # radius

    for i in range(k, h - k):
        for j in range(k, w - k):

            # extract neighborhood patch
            patch = img[i - k:i + k + 1, j - k:j + k + 1]

            # apply custom rule
            if kernel_func(patch):
                out[i, j] = 0

    return out
