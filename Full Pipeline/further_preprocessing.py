"""Helper routines for removing axes, labels, and circular artifacts from images."""

import cv2
import numpy as np

from utils import * # open_image


def remove_mask(image, mask):
    """Black out every pixel selected by ``mask``."""
    img = open_image(image)

    print(img.shape)
    if len(img.shape) == 2:
        img[mask == 255] = 0
    else:
        img[mask == 255] = [0, 0, 0]

    return img

def remove_outside_axes(image, def_points):
    """Keep only the rectangular work area bounded by the detected axes."""
    img = open_image(image)

    x1, y1, x2, y2 = def_points
    h, w = img.shape[:2]
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    mask = np.zeros_like(img)
    mask[y1:y2, x1:x2] = img[y1:y2, x1:x2]

    return mask

def remove_axes(image, remove=True, lower_bound=np.array([140, 50, 50]), upper_bound=np.array([170, 255, 255])):
    """Detect the magenta axes and optionally paint them out."""
    img = open_image(image)

    mask = create_color_mask_HSV(img)

    if remove:
        img = remove_mask(img, mask)

    return img, mask

def remove_axes_label(image, axis_points, height=30, width=80):
    """Inpaint the small axis annotation label near the vertical axis."""
    img = open_image(image)

    x1, y1, x2, y2 = axis_points
    y1, y2 = min(y1, y2), max(y1, y2)
    x_left_center = int(np.mean((x1,x2)))+3  # x-coordinate of center of left side
    y_left_center = y1 + int((y2-y1)/2) - 7 # y-coordinate of center of left side
    w = width  # width of rectangle
    h = height  # height of rectangle

    x1 = int(x_left_center)
    y1 = int(y_left_center - h / 2)

    x2 = int(x_left_center + w)
    y2 = int(y_left_center + h / 2)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return result

def extract_circular_patch(image, x, y, radius, zoom=8):
    """Extract and display an integer-centered circular patch for manual inspection."""
    img = open_image(image)

    h, w = img.shape[:2]

    if (x - radius < 0 or x + radius >= w or
        y - radius < 0 or y + radius >= h):
        raise ValueError("Patch goes out of image bounds")

    patch = img[y-radius:y+radius+1, x-radius:x+radius+1].copy()

    mask = np.zeros((2*radius+1, 2*radius+1), dtype=np.uint8)
    cv2.circle(mask, (radius, radius), radius, 255, -1)

    if len(patch.shape) == 3:  # color
        masked_patch = cv2.bitwise_and(patch, patch, mask=mask)
    else:  # grayscale
        masked_patch = cv2.bitwise_and(patch, patch, mask=mask)

    zoomed = cv2.resize(
        masked_patch,
        None,
        fx=zoom,
        fy=zoom,
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imshow("circular patch", zoomed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return masked_patch


def extract_circular_patch_subpixel(image, x, y, radius, zoom=8):
    """
    Extract and display a circular patch around a subpixel-accurate center.
    """
    x, y = float(x), float(y)

    img = open_image(image)
    size = 2 * radius + 1

    patch = cv2.getRectSubPix(
        img,
        (size, size),   # width, height
        (x, y)          # FLOAT center
    )

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (radius, radius), radius, 255, -1)

    if len(patch.shape) == 3:
        masked = cv2.bitwise_and(patch, patch, mask=mask)
    else:
        masked = cv2.bitwise_and(patch, patch, mask=mask)

    zoomed = cv2.resize(masked, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    zoomed = cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY)

    cv2.imshow("subpixel patch", zoomed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    return masked

def create_circle_mask(shape, centers, radius=6):
    """Build a mask that covers a fixed-radius disk around every detected circle."""
    mask = np.zeros(shape[:2], dtype=np.uint8)

    for (x, y) in centers:
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(mask, (cx, cy), radius, 255, -1)

    return mask

def inpaint_circles(img, centers, radius=6):
    """Remove detected circles with Telea inpainting."""
    img = open_image(img)
    mask = create_circle_mask(img.shape, centers, radius)

    result = cv2.inpaint(
        img,
        mask,
        inpaintRadius=2,
        flags=cv2.INPAINT_TELEA
    )

    return result

def mask_circles(img, centers, radius=6):
    """
    Black out circles given in ``(x, y)`` convention.
    """

    output = open_image(img)

    for (x, y) in centers:
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(output, (cx, cy), radius, 0, -1)  # fill with black


    return output

def mask_circles_yx(img, centers, radius=8):
    """
    Black out circles given in ``(y, x)`` convention.
    """

    output = open_image(img)

    for (y, x) in centers:
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(output, (cx, cy), radius, 0, -1)  # fill with black


    return output

def mask_circles_yx_enhanced(img, centers, radius=8, white_threshold=1):

    """

    Remove only circles whose interior still contains bright pixels.

    Parameters

    ----------

    img : ndarray

        Input grayscale or BGR image.

    centers : list of (y, x)

        Circle centers in (y, x) convention.

    radius : int

        Radius of circle to test/remove.

    white_threshold : int

        Pixel value considered white (> threshold).

        For binary images use 1.

        For uint8 grayscale use e.g. 127.

    Returns

    -------

    output : ndarray

        Image with accepted circles blacked out.

    removed_centers : list of (y, x)

        Only circles that were actually removed.

    """

    output = open_image(img)

    removed_centers = []

    if len(output.shape) == 3:
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    else:
        gray = output.copy()

    h, w = gray.shape[:2]

    for (y, x) in centers:

        cx = int(round(x))

        cy = int(round(y))

        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, -1)

        inside_pixels = gray[mask > 0]

        if np.any(inside_pixels >= white_threshold):
            cv2.circle(output, (cx, cy), radius, 0, -1)
            removed_centers.append((y, x))

    return output, removed_centers


def collapse_circle_to_point(img, centers, radius=6, search_radius=2, black_thresh=30):
    """
    Heuristic repair: connect dark ring pixels back to a circle center.
    """

    output = img.copy()

    h, w = img.shape[:2]

    r_outer = radius + search_radius
    r2_inner = radius * radius
    r2_outer = r_outer * r_outer

    for (x, y) in centers:
        cx, cy = int(round(x)), int(round(y))

        x0 = max(0, cx - r_outer)
        x1 = min(w, cx + r_outer + 1)
        y0 = max(0, cy - r_outer)
        y1 = min(h, cy + r_outer + 1)

        for yy in range(y0, y1):
            dy2 = (yy - cy) ** 2
            for xx in range(x0, x1):
                dx2 = (xx - cx) ** 2
                dist2 = dx2 + dy2

                if r2_inner < dist2 <= r2_outer:
                    if img[yy, xx] <= black_thresh:
                        cv2.line(output, (xx, yy), (cx, cy), 255, 1)

    return output
