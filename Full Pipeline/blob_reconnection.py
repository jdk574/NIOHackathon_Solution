"""Reconnect short local branches around masked circular blobs."""

import numpy as np
import cv2

from nodes_classification import (
    extract_branches_inside_radius,
)


def point_to_center_distance(p, center):
    """
    Euclidean distance between two points p=(y,x), center=(y,x).
    """
    return float(np.hypot(p[0] - center[0], p[1] - center[1]))


def select_branch_connection_point(branch, center, mode="closest_pixel"):
    """
    Select which point on a surviving branch should reconnect to the blob center.

    Parameters
    ----------
    branch : dict
        One branch dict returned by extract_branches_inside_radius
    center : tuple
        (y, x)
    mode : str
        One of:
            - 'path_start'
            - 'closest_pixel'
            - 'endpoint_near_center'

    Returns
    -------
    p : tuple or None
        Selected point (y, x)
    """
    path = branch["path"]
    pixels = branch["pixels"]
    endpoints = branch["endpoints"]

    if len(path) == 0 and len(pixels) == 0:
        return None

    if mode == "path_start":
        if len(path) > 0:
            return path[0]
        pts = np.array(pixels, dtype=int)
        d2 = (pts[:, 0] - center[0]) ** 2 + (pts[:, 1] - center[1]) ** 2
        return tuple(pts[np.argmin(d2)])

    elif mode == "closest_pixel":
        pts = np.array(pixels, dtype=int)
        d2 = (pts[:, 0] - center[0]) ** 2 + (pts[:, 1] - center[1]) ** 2
        return tuple(pts[np.argmin(d2)])

    elif mode == "endpoint_near_center":
        if len(endpoints) > 0:
            pts = np.array(endpoints, dtype=int)
            d2 = (pts[:, 0] - center[0]) ** 2 + (pts[:, 1] - center[1]) ** 2
            return tuple(pts[np.argmin(d2)])
        elif len(path) > 0:
            return path[0]
        else:
            pts = np.array(pixels, dtype=int)
            d2 = (pts[:, 0] - center[0]) ** 2 + (pts[:, 1] - center[1]) ** 2
            return tuple(pts[np.argmin(d2)])

    else:
        raise ValueError("mode must be 'path_start', 'closest_pixel', or 'endpoint_near_center'")


def draw_connection_to_center(img, p, center, value=255, thickness=1):
    """
    Draw a line from ``p`` back to the blob center on the working skeleton.
    """
    if p is None:
        return img

    y0, x0 = p
    y1, x1 = center
    cv2.line(img, (x0, y0), (x1, y1), value, thickness)
    return img


def connect_local_branches_to_blob_center(
    skeleton,
    center,
    outer_radius=20,
    inner_radius=5,
    min_branch_length=3,
    connect_from="closest_pixel",
    min_path_length=0,
    max_connection_gap=None,
    require_near_inner_boundary=False,
    inner_boundary_tol=3.0,
    line_value=255,
    line_thickness=1,
    return_debug=False,
):
    """
    Reconnect local branch fragments around one masked blob center.

    Parameters
    ----------
    skeleton : np.ndarray
        Binary or uint8 skeleton image.
    center : tuple
        Blob center as (y, x).
    outer_radius : int
    inner_radius : int
    min_branch_length : int
        Minimum connected-component size to keep.
    connect_from : str
        'path_start', 'closest_pixel', or 'endpoint_near_center'
    min_path_length : int
        Reject branches whose ordered path is shorter than this.
    max_connection_gap : float or None
        If given, reject branch connection points farther than this from center.
    require_near_inner_boundary : bool
        If True, only keep branches whose selected connection point lies near
        the inner cut boundary.
    inner_boundary_tol : float
        Allowed distance from the ideal inner boundary radius.
    line_value : int
    line_thickness : int
    return_debug : bool

    Returns
    -------
    repaired : np.ndarray
        Skeleton with accepted local branches connected to center.
    result : dict   (only if return_debug=True)
    """
    if skeleton.ndim != 2:
        raise ValueError("skeleton must be a 2D array")

    cy, cx = int(round(center[0])), int(round(center[1]))
    center = (cy, cx)

    skel_bin = (skeleton > 0).astype(np.uint8)

    # Look only at the ring around the blob so the reconnection decision stays local.
    branches, kept_local = extract_branches_inside_radius(
        skeleton=skel_bin,
        center=center,
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        min_branch_length=min_branch_length,
    )

    repaired = (skel_bin * 255).astype(np.uint8)

    accepted_branches = []
    rejected_branches = []
    connected_points = []

    for b in branches:
        branch_info = dict(b)

        if len(branch_info["path"]) < min_path_length:
            branch_info["accepted"] = False
            branch_info["reject_reason"] = (
                f"path too short ({len(branch_info['path'])} < {min_path_length})"
            )
            rejected_branches.append(branch_info)
            continue

        p = select_branch_connection_point(branch_info, center, mode=connect_from)
        if p is None:
            branch_info["accepted"] = False
            branch_info["reject_reason"] = "no valid connection point found"
            rejected_branches.append(branch_info)
            continue

        dist_c = point_to_center_distance(p, center)
        branch_info["connection_point"] = p
        branch_info["connection_distance"] = dist_c

        if max_connection_gap is not None and dist_c > max_connection_gap:
            branch_info["accepted"] = False
            branch_info["reject_reason"] = (
                f"connection point too far from center ({dist_c:.2f} > {max_connection_gap})"
            )
            rejected_branches.append(branch_info)
            continue

        if require_near_inner_boundary:
            boundary_err = abs(dist_c - inner_radius)
            branch_info["inner_boundary_error"] = boundary_err
            if boundary_err > inner_boundary_tol:
                branch_info["accepted"] = False
                branch_info["reject_reason"] = (
                    f"connection point not near inner boundary "
                    f"(error {boundary_err:.2f} > {inner_boundary_tol})"
                )
                rejected_branches.append(branch_info)
                continue

        repaired = draw_connection_to_center(
            repaired,
            p=p,
            center=center,
            value=line_value,
            thickness=line_thickness,
        )

        branch_info["accepted"] = True
        accepted_branches.append(branch_info)
        connected_points.append(p)

    result = {
        "center": center,
        "num_branches": len(branches),
        "num_accepted": len(accepted_branches),
        "num_rejected": len(rejected_branches),
        "accepted_branches": accepted_branches,
        "rejected_branches": rejected_branches,
        "connected_points": connected_points,
        "kept_local": kept_local,
        "connect_from": connect_from,
        "inner_radius": inner_radius,
        "outer_radius": outer_radius,
    }

    if return_debug:
        return repaired, result
    return repaired


def connect_all_blob_centers(
    skeleton,
    blob_centers,
    outer_radius=20,
    inner_radius=5,
    min_branch_length=3,
    connect_from="closest_pixel",
    min_path_length=0,
    max_connection_gap=None,
    require_near_inner_boundary=False,
    inner_boundary_tol=3.0,
    line_value=255,
    line_thickness=1,
    round_input_centers=True,
    return_debug=False,
):
    """
    Apply local blob-center reconnection for every detected circle center.

    Parameters
    ----------
    skeleton : np.ndarray
        Binary or uint8 skeleton image.
    blob_centers : iterable
        Iterable of centers as (y, x).
    outer_radius : int
    inner_radius : int
    min_branch_length : int
    connect_from : str
    min_path_length : int
    max_connection_gap : float or None
    require_near_inner_boundary : bool
    inner_boundary_tol : float
    line_value : int
    line_thickness : int
    round_input_centers : bool
    return_debug : bool

    Returns
    -------
    repaired : np.ndarray
    all_results : list[dict]   (only if return_debug=True)
    """
    repaired = (skeleton > 0).astype(np.uint8) * 255
    all_results = []

    h, w = repaired.shape[:2]

    for c in blob_centers:
        y, x = c

        if round_input_centers:
            center = (int(round(y)), int(round(x)))
        else:
            center = (int(y), int(x))

        if not (0 <= center[0] < h and 0 <= center[1] < w):
            all_results.append({
                "center": center,
                "skipped": True,
                "reason": "center outside image bounds",
            })
            continue

        repaired, result = connect_local_branches_to_blob_center(
            skeleton=repaired,
            center=center,
            outer_radius=outer_radius,
            inner_radius=inner_radius,
            min_branch_length=min_branch_length,
            connect_from=connect_from,
            min_path_length=min_path_length,
            max_connection_gap=max_connection_gap,
            require_near_inner_boundary=require_near_inner_boundary,
            inner_boundary_tol=inner_boundary_tol,
            line_value=line_value,
            line_thickness=line_thickness,
            return_debug=True,
        )

        result["skipped"] = False
        all_results.append(result)

    if return_debug:
        return repaired, all_results
    return repaired


if __name__ == "__main__":
    # Example usage
    # blob_centers must be in (y, x) convention
    import matplotlib.pyplot as plt

    # Dummy example skeleton
    skeleton = np.zeros((200, 200), dtype=np.uint8)

    # Example broken branches around a blob center
    cv2.line(skeleton, (100, 40), (100, 80), 255, 1)
    cv2.line(skeleton, (100, 120), (100, 160), 255, 1)
    cv2.line(skeleton, (40, 100), (80, 100), 255, 1)
    cv2.line(skeleton, (120, 100), (160, 100), 255, 1)

    blob_centers = [(100, 100)]

    repaired, debug = connect_all_blob_centers(
        skeleton,
        blob_centers,
        outer_radius=30,
        inner_radius=12,
        min_branch_length=3,
        connect_from="closest_pixel",
        min_path_length=3,
        max_connection_gap=20,
        require_near_inner_boundary=False,
        return_debug=True,
    )

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(skeleton, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(repaired, cmap="gray")
    plt.title("Repaired")
    plt.axis("off")

    plt.show()

    print(debug)
