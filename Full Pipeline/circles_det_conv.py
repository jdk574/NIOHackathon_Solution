import cv2
import numpy as np
from pathlib import Path

"""Template-matching stage used to find repeated circular artifacts in the drawing."""

from utils import open_image

# Saved offline template used by the runtime pipeline.
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "template_99_in.npy"
template_default = np.load(TEMPLATE_PATH)


# -------------------------------
# 1. Normalize + average patches
# -------------------------------
def build_template(patches):
    """Build a normalized average template from manually sampled circular patches."""

    if patches is None:
        print('No patches')
        return None
    norm_patches = []

    for p in patches:
        p = p.astype(np.float32)
        p -= p.mean()
        p /= (p.std() + 1e-6)
        norm_patches.append(p)

    stack = np.stack(norm_patches, axis=0)
    template = np.mean(stack, axis=0)

    # normalize final template
    template -= template.mean()
    template /= (template.std() + 1e-6)

    return template


# -------------------------------
# 2. Template matching
# -------------------------------
def match_template(img, template):
    """Run normalized correlation matching between the image and the circle template."""
    img = img.astype(np.float32)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(
        img,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    return result


# -------------------------------
# 3. Subpixel peak refinement
# -------------------------------
def refine_subpixel(response, x, y):
    """
    Quadratic refinement using 3x3 neighborhood
    """
    h, w = response.shape

    if x <= 0 or x >= w-1 or y <= 0 or y >= h-1:
        return float(x), float(y)

    # neighborhood
    patch = response[y-1:y+2, x-1:x+2]

    # gradients
    dx = 0.5 * (patch[1,2] - patch[1,0])
    dy = 0.5 * (patch[2,1] - patch[0,1])

    dxx = patch[1,2] - 2*patch[1,1] + patch[1,0]
    dyy = patch[2,1] - 2*patch[1,1] + patch[0,1]

    # avoid division by zero
    if dxx == 0 or dyy == 0:
        return float(x), float(y)

    sub_x = x - dx / dxx
    sub_y = y - dy / dyy

    return float(sub_x), float(sub_y)


# -------------------------------
# 4. Detect peaks + refine
# -------------------------------
def detect_subpixel_centers(response, template, threshold=0.7, min_dist=8):
    """Threshold, suppress duplicates, and refine candidate circle centers."""
    points = np.argwhere(response >= threshold)

    h, w = template.shape[:2]
    cx_offset = w // 2
    cy_offset = h // 2

    points = [(int(p[1] + cx_offset), int(p[0] + cy_offset)) for p in points]

    # simple non-max suppression
    filtered = []
    for p in points:
        keep = True
        for q in filtered:
            if np.linalg.norm(np.array(p) - np.array(q)) < min_dist:
                keep = False
                break
        if keep:
            filtered.append(p)

    # refine to subpixel
    refined = []
    for (x, y) in filtered:
        sx, sy = refine_subpixel(response, x - w // 2, y - h // 2)

        # then shift again to center
        sx += w / 2
        sy += h / 2
        refined.append((sx, sy))

    return refined


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def find_circles(image, patches=None, template=template_default):
    """High-level helper used by preprocessing to locate circles in one call."""
    img = open_image(image)

    if template is None:
        template = build_template(patches)

    response = match_template(img, template)

    centers = detect_subpixel_centers(response, template)

    centers_yx = [(y, x) for (x, y) in centers]

    return centers_yx, template, response
