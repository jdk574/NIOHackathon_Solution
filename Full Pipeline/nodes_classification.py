"""Validate candidate graph nodes by inspecting the branch directions around them."""

import numpy as np
import cv2

NEIGHBORS_8 = [
    (-1, -1), (-1,  0), (-1,  1),
    ( 0, -1),           ( 0,  1),
    ( 1, -1), ( 1,  0), ( 1,  1),
]


def circular_mask(shape, center, radius):
    """
    Binary circular mask centered at (y, x).
    Returns uint8 mask with values 0 or 1.
    """
    h, w = shape
    cy, cx = center
    Y, X = np.ogrid[:h, :w]
    mask = (Y - cy) ** 2 + (X - cx) ** 2 <= radius ** 2
    return mask.astype(np.uint8)


def get_neighbors_8(p, img_bin):
    """
    Return active 8-neighbors of p=(y, x) in binary image img_bin.
    """
    y, x = p
    h, w = img_bin.shape
    out = []
    for dy, dx in NEIGHBORS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and img_bin[ny, nx]:
            out.append((ny, nx))
    return out


def connected_components_points(binary_img):
    """
    Return connected components as lists of (y, x) points.
    """
    nlabels, labels = cv2.connectedComponents(binary_img.astype(np.uint8), connectivity=8)
    comps = []
    for lab in range(1, nlabels):
        pts = np.argwhere(labels == lab)
        comps.append([tuple(p) for p in pts])
    return comps


def order_component_path(component, img_bin):
    """
    Order a skeleton component into a path.
    Best for branch-like components.

    Returns
    -------
    path : list[(y, x)]
    endpoints : list[(y, x)]
    """
    comp_set = set(component)

    def comp_neighbors(p):
        return [q for q in get_neighbors_8(p, img_bin) if q in comp_set]

    # A branch component is usually traced from one endpoint toward the blob center.
    endpoints = [p for p in component if len(comp_neighbors(p)) == 1]

    if len(endpoints) >= 1:
        start = endpoints[0]
    else:
        start = component[0]

    path = [start]
    visited = {start}
    prev = None
    cur = start

    while True:
        nbrs = comp_neighbors(cur)
        if prev is not None:
            nbrs = [q for q in nbrs if q != prev]
        nbrs = [q for q in nbrs if q not in visited]

        if len(nbrs) == 0:
            break

        nxt = nbrs[0]
        path.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt

    return path, endpoints


def dist2(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def path_start_near_center(path, center):
    """
    Make path start at the end closer to center.
    """
    if len(path) <= 1:
        return path
    if dist2(path[0], center) <= dist2(path[-1], center):
        return path
    return path[::-1]


def estimate_branch_angle_from_center(path, center, step=5):
    """
    Estimate branch angle using a point a few pixels away from the center-side
    of the branch.

    Returns angle in [0, 180), so opposite directions are treated as the same line.
    """
    if len(path) < 2:
        return None

    i = min(step, len(path) - 1)
    p0 = np.array(center, dtype=float)
    p1 = np.array(path[i], dtype=float)

    v = p1 - p0
    norm = np.linalg.norm(v)
    if norm == 0:
        return None

    dy = v[0]
    dx = v[1]
    angle = np.degrees(np.arctan2(dy, dx)) % 180.0
    return angle


def line_angle_difference_deg(a, b):
    """
    Difference between two undirected line angles in degrees.
    Result is in [0, 90].
    """
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def extract_branches_inside_radius(skeleton, center, outer_radius, inner_radius, min_branch_length=2):
    """
    Remove the center blob, keep only the nearby skeleton ring, and extract branches.

    Returns
    -------
    branches : list[dict]
        Each dict has:
        {
            'component': int,
            'pixels': [...],
            'path': [...],          # ordered, starting near center
            'endpoints': [...],
        }
    kept : np.ndarray
        uint8 image 0/255 of kept local skeleton after center removal
    """
    skel_bin = (skeleton > 0).astype(np.uint8)

    outer = circular_mask(skel_bin.shape, center, outer_radius)
    inner = circular_mask(skel_bin.shape, center, inner_radius)

    kept = skel_bin * outer
    kept[inner > 0] = 0

    components = connected_components_points(kept)

    branches = []
    for i, comp in enumerate(components):
        if len(comp) < min_branch_length:
            continue

        path, endpoints = order_component_path(comp, kept)
        path = path_start_near_center(path, center)

        branches.append({
            "component": i,
            "pixels": comp,
            "path": path,
            "endpoints": endpoints,
        })

    return branches, (kept * 255).astype(np.uint8)


def classify_node_by_branch_angle_coincidence(
    skeleton,
    center,
    outer_radius=20,
    inner_radius=4,
    direction_step=5,
    angle_tol_deg=15,
    min_branch_length=3,
    min_branches_for_node=3,
    return_debug=False,
):
    """
    Decide whether a candidate point behaves like a real node.

    The local skeleton around the candidate is split into outgoing branches.
    If every surviving branch points in essentially the same direction, the
    candidate is treated as part of a line rather than a true junction. A
    candidate becomes a node once at least one branch direction disagrees with
    the rest.
    """
    branches, kept_local = extract_branches_inside_radius(
        skeleton=skeleton,
        center=center,
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        min_branch_length=min_branch_length,
    )

    valid_branches = []
    angles = []
    for b in branches:
        ang = estimate_branch_angle_from_center(
            path=b["path"],
            center=center,
            step=direction_step,
        )
        if ang is None:
            continue
        b2 = dict(b)
        b2["angle_deg"] = ang
        valid_branches.append(b2)
        angles.append(ang)

    n = len(angles)
    coincidence_matrix = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            coincidence_matrix[i, j] = line_angle_difference_deg(angles[i], angles[j])

    if n < min_branches_for_node:
        result = {
            "is_node": False,
            "reason": f"only {n} valid branch directions; fewer than required {min_branches_for_node}",
            "center": center,
            "num_branches": len(branches),
            "num_valid_angles": n,
            "branch_angles_deg": angles,
            "branches": valid_branches,
            "coincidence_matrix_deg": coincidence_matrix,
            "kept_local": kept_local,
        }
        if return_debug:
            return result
        result.pop("branches")
        result.pop("coincidence_matrix_deg")
        result.pop("kept_local")
        return result

    ref = angles[0]
    all_coincide = all(line_angle_difference_deg(ref, a) <= angle_tol_deg for a in angles[1:])

    if all_coincide:
        reason = "all local branch directions coincide"
        is_node = False
    else:
        reason = "at least one local branch direction does not coincide with the others"
        is_node = True

    result = {
        "is_node": is_node,
        "reason": reason,
        "center": center,
        "num_branches": len(branches),
        "num_valid_angles": n,
        "branch_angles_deg": angles,
        "branches": valid_branches,
        "coincidence_matrix_deg": coincidence_matrix,
        "kept_local": kept_local,
    }

    if return_debug:
        return result

    result.pop("branches")
    result.pop("coincidence_matrix_deg")
    result.pop("kept_local")
    return result


def classify_all_nodes(
    skeleton,
    nodes,
    outer_radius=20,
    inner_radius=4,
    direction_step=5,
    angle_tol_deg=15,
    min_branch_length=3,
    min_branches_for_node=2,
    round_input_nodes=True,
    return_debug=False,
):
    """
    Run the branch-direction validation stage for every candidate node.

    Parameters
    ----------
    skeleton : np.ndarray
        Binary or uint8 skeleton image
    nodes : array-like
        Iterable of node coordinates. Each node should be (y, x).
        Float coordinates are allowed; they will be rounded if round_input_nodes=True.
    outer_radius : float
    inner_radius : float
    direction_step : int
    angle_tol_deg : float
    min_branch_length : int
    min_branches_for_node : int
    round_input_nodes : bool
        If True, round nodes to integer pixel coordinates
    return_debug : bool
        If True, keep branch details and local masks for each node

    Returns
    -------
    confirmed_nodes : list[(y, x)]
        Nodes classified as real nodes
    rejected_nodes : list[(y, x)]
        Nodes rejected by the angle coincidence test
    all_results : list[dict]
        Full result per node
    """
    confirmed_nodes = []
    rejected_nodes = []
    all_results = []

    h, w = skeleton.shape[:2]

    for node in nodes:
        y, x = node

        if round_input_nodes:
            center = (int(round(y)), int(round(x)))
        else:
            center = (int(y), int(x))

        # skip nodes outside image
        if not (0 <= center[0] < h and 0 <= center[1] < w):
            result = {
                "is_node": False,
                "reason": "node outside image bounds",
                "center": center,
            }
            all_results.append(result)
            rejected_nodes.append(center)
            continue

        result = classify_node_by_branch_angle_coincidence(
            skeleton=skeleton,
            center=center,
            outer_radius=outer_radius,
            inner_radius=inner_radius,
            direction_step=direction_step,
            angle_tol_deg=angle_tol_deg,
            min_branch_length=min_branch_length,
            min_branches_for_node=min_branches_for_node,
            return_debug=return_debug,
        )

        all_results.append(result)

        if result["is_node"]:
            confirmed_nodes.append(center)
        else:
            rejected_nodes.append(center)

    return confirmed_nodes, rejected_nodes, all_results
