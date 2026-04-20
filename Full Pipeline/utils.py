import os, cv2
import numpy as np
from pathlib import Path

"""Shared small utilities for image loading, coordinate conversion, and masks."""


def open_image(image):
    """Accept either a path or an image array and always return an OpenCV image."""
    if isinstance(image, str) or isinstance(image, Path):
        if not os.path.isfile(image):
            raise ValueError("Invalid file path")
        img = cv2.imread(image)
        if img is None:
            raise ValueError("File is not a valid image")
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise TypeError("Input must be a file path or an OpenCV image (numpy array)")

    return img


def create_color_mask_HSV(image, lower_bound=np.array([140, 50, 50]), upper_bound=np.array([170, 255, 255])):

    '''
    Default bounds are for pink axes

    :param image:
    :param lower_bound: in HSV color space
    :param upper_bound: in HSV color space
    :return:
    '''

    # Open Image
    img = open_image(image)

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    return mask

def filter_nodes_keep_indices(nodes_array, node_map):
    """
    Remove entries from node_map whose coordinates are not in nodes_array,
    while keeping the original indices of the remaining nodes.

    Parameters
    ----------
    nodes_array : list of tuple
        List of node coordinates, e.g. [(x1, y1), (x2, y2), ...]
    node_map : dict
        Dictionary with indices as keys and coordinates as values
        e.g. {0: (10, 20), 1: (30, 40), 2: (50, 60)}

    Returns
    -------
    filtered_map : dict
        Same structure as node_map, but only nodes present in nodes_array remain.
        Original indices are preserved.
    """

    node_set = {tuple(n) for n in nodes_array}

    filtered_map = {
        idx: coord
        for idx, coord in node_map.items()
        if tuple(coord) in node_set
    }

    return filtered_map

def custom_bgrtogray_transform(image, weights = [[0.34, 0.26, 0.40]]):
    """Weighted color-to-gray transform used during experimentation."""
    img = open_image(image)

    M = np.array(weights, dtype=np.float32)

    gray = cv2.transform(img, M)

    gray = gray.squeeze().astype(np.uint8)

    return gray

def turn_green_to_black(image, lower=np.array([40, 80, 80]), upper=np.array([85, 255, 255])):
    """Suppress green overlays that are not part of the actual mesh geometry."""
    img = open_image(image)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # turn masked pixels black
    img[mask > 0] = (0, 0, 0)

    return img
