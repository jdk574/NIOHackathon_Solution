import cv2
import numpy as np
import os
"""Detect the magenta axes and recover the image-to-model scaling."""

from extract_center_line import apply_custom_kernel_rule
from utils import *


def edge_det(mask, h):
    """Clean the axis mask and extract strong edges before Hough detection."""
    k = max(3, h // 500)
    kernel = np.ones((k, k), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Edge detection
    edges = cv2.Canny(mask, 50, 150)

    return edges

def binarization(gray):
    """Binarize the magenta axis mask so line detection is more stable."""
    # Optional: denoise (helps adaptive threshold stability)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        7,  # block size (must be odd)
        2  # constant subtracted from mean
    )

    kernel = np.ones((4, 4), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    new_binary = cv2.adaptiveThreshold(
        dilated,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        7,  # block size (must be odd)
        2  # constant subtracted from mean
    )

    return new_binary

def preprocess(image_path):
    """Create a simple HSV mask of the magenta axes."""
    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    # 2. Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. Threshold pink/purple
    lower = np.array([140, 50, 50])
    upper = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    gray = mask.copy()

    return gray

def extract_axes(image_path, edges, debug=False):
    """Recover the two axes, their intersection, and the scaling factors."""
    # # 1. Load image
    img = open_image(image_path)
    #
    h, w = img.shape[:2]
    #

    # 6. Hough line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=w // 4,
        maxLineGap=6
    )

    if lines is None:
        raise ValueError("No lines detected")

    # 7. Separate horizontal and vertical lines. The competition images contain
    # one dominant horizontal magenta axis and one dominant vertical magenta axis.
    horizontal_axis = ()
    vertical_axis = ()

    for line in lines:
        x1, y1, x2, y2 = line[0]

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dx > dy:  # horizontal
            horizontal_axis = (x1, y1, x2, y2)
        else:        # vertical
            vertical_axis = (x1, y1, x2, y2)

    if not horizontal_axis or not vertical_axis:
        raise ValueError("Could not find both axis lines")

    top_line = list(horizontal_axis)
    right_line = list(vertical_axis)

    top_line[0] = horizontal_axis[0] - 2
    top_line[2] = horizontal_axis[2] + 1

    right_line[0] = vertical_axis[0] - 1
    right_line[1] = vertical_axis[1] + 1
    right_line[2] = vertical_axis[2] - 1
    right_line[3] = vertical_axis[3] - 1

    top_line = tuple(top_line)
    right_line = tuple(right_line)

    print(top_line)
    print(right_line)


    # 9. Compute the origin as the axes intersection in image coordinates.
    def intersect(l1, l2):
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2

        A1 = y2 - y1
        B1 = x1 - x2
        C1 = A1 * x1 + B1 * y1

        A2 = y4 - y3
        B2 = x3 - x4
        C2 = A2 * x3 + B2 * y3

        det = A1 * B2 - A2 * B1
        if det == 0:
            return None

        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det

        return int(x), int(y)

    if top_line[0] == right_line[0] and top_line[1] == right_line[1]:
        intersection = (top_line[0], top_line[1])
        print('On top')
    else:
        intersection = intersect(horizontal_axis, vertical_axis)

    if intersection is None:
        raise ValueError("Lines do not intersect")

    work_area = (top_line[2], top_line[3], right_line[2], right_line[3])


    # 10. Estimate coordinate bounds from the detected axes themselves.
    # The horizontal axis gives the rightmost x extent and the vertical
    # axis gives the upper y extent, both measured from the same origin.
    x_axis_end = max(top_line[0], top_line[2])
    y_axis_top = min(right_line[1], right_line[3])

    origin_x, origin_y = intersection
    x_span = float(x_axis_end - origin_x)
    y_span = float(origin_y - y_axis_top)

    if x_span <= 0 or y_span <= 0:
        raise ValueError(
            f"Invalid axis spans detected: x_span={x_span}, y_span={y_span}, "
            f"intersection={intersection}, top_line={top_line}, right_line={right_line}"
        )

    x_scale = 100.0 / x_span
    y_scale = 75.0 / y_span


    vis = img.copy()



    # draw selected axes
    cv2.line(vis, (top_line[0], top_line[1]), (top_line[2], top_line[3]), (0,255,0), 1)
    cv2.line(vis, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (255,0,0), 1)

    # draw intersection
    cv2.circle(vis, intersection, 1, (0,0,255))

    if debug:
        name = os.path.splitext(os.path.basename(image_path))[0]
        filename = name+"_hough.jpg"
        cv2.imshow("debug", vis)
        cv2.imwrite(filename, vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "top_line": top_line,
        "right_line": right_line,
        "intersection": intersection,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "work_area": work_area,
        "vertical_axis": right_line,
        "horizontal_axis": top_line,
        "visualization": vis
    }


def run_axes(image_path):
    """Public entrypoint used by the rest of the pipeline for axis calibration."""
    gray = create_color_mask_HSV(image_path)
    binary = binarization(gray)
    out = apply_custom_kernel_rule(binary)
    result = extract_axes(image_path, out)

    return result

















