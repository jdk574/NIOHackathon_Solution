import cv2
import numpy as np
from skimage.morphology import dilation
from sklearn.cluster import DBSCAN

"""Detect, cluster, and refine graph node positions on the skeleton image."""

from edge_detection import *
from utils import *
from visualization import *
from preprocessing import preprocess_and_skeletonize


def find_skeleton_nodes(skeleton):
    """
    skel: binary skeleton image (0 or 255)
    returns:
        endpoints: list of (x, y)
        junctions: list of (x, y)
    """

    h, w = skeleton.shape
    endpoints = []
    junctions = []
    all_nodes = []
    node_map = {}

    # 8-neighborhood kernel (exclude center)
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    # Count neighbors using convolution
    neighbor_count = cv2.filter2D(skeleton, -1, kernel)

    for y in range(1, h-1):
        for x in range(1, w-1):
            if skeleton[y, x] == 1:
                n = neighbor_count[y, x]

                if n == 1:
                    endpoints.append((x, y))
                    all_nodes.append((x, y))
                    node_map[(x, y)] = len(all_nodes)
                elif n >= 3:
                    junctions.append((x, y))
                    all_nodes.append((x, y))
                    node_map[(x, y)] = len(all_nodes)

    return endpoints, junctions, all_nodes, node_map


def cluster_dbscan(points, eps=8, min_samples=2, snap_to_int=True):
    """Cluster nearby raw node detections into one node hypothesis per location."""
    if len(points) == 0:
        return [], {}

    pts = np.asarray(points, dtype=float)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = clustering.labels_

    clustered = []
    cl_node_map = {}

    # First handle true clusters
    valid_labels = sorted(lab for lab in set(labels) if lab != -1)

    for label in valid_labels:
        cluster_pts = pts[labels == label]
        center = cluster_pts.mean(axis=0)

        if snap_to_int:
            coor = tuple(np.rint(center).astype(int))
        else:
            coor = tuple(center)

        if coor not in cl_node_map:
            cl_node_map[coor] = len(clustered)
            clustered.append(coor)

    # Then keep noise points as single nodes
    noise_pts = pts[labels == -1]
    for p in noise_pts:
        if snap_to_int:
            coor = tuple(np.rint(p).astype(int))
        else:
            coor = tuple(p)

        if coor not in cl_node_map:
            cl_node_map[coor] = len(clustered)
            clustered.append(coor)

    return clustered, cl_node_map


def snap_points_bruteforce(skeleton, points):
    """Snap clustered node hypotheses back onto the nearest white skeleton pixel."""
    ys, xs = np.where(skeleton > 0)
    skel_points = np.stack([xs, ys], axis=1)

    snapped = []

    for p in points:
        dists = np.linalg.norm(skel_points - p, axis=1)
        snapped.append(skel_points[np.argmin(dists)])

    return np.array(snapped)

def has_neighbors_from_graph(node_index, adjacency_list):
    return len(adjacency_list[node_index]) > 0

def validate_nodes(nodes, skeleton, adjacency_list, neighbor_radius=1.5, white_threshold=1):
    """Optional local sanity check for whether nodes still touch skeleton pixels."""
    """
    nodes: array-like of shape (N, 2) -> [(x, y), ...]
    skeleton: 2D numpy array (grayscale image)
    neighbor_radius: distance threshold to consider nodes as neighbors
    white_threshold: pixel intensity threshold for "white"

    Returns:
        valid_mask: list of booleans per node
        issues: list of dicts describing problems
    """

    nodes = np.array(nodes)
    h, w = skeleton.shape

    valid_mask = []
    issues = []

    for i, (x, y) in enumerate(nodes):
        node_ok = True
        node_issues = []

        xi, yi = int(round(x)), int(round(y))

        # --- Check 1: inside bounds ---
        if not (0 <= xi < w and 0 <= yi < h):
            node_ok = False
            node_issues.append("out_of_bounds")
        else:
            # --- Check 2: on white pixel ---
            if skeleton[yi, xi] < white_threshold:
                node_ok = False
                node_issues.append("not_on_white")

        # --- Check 3: has neighbors ---
        neighbours = has_neighbors_from_graph(i, adjacency_list)

        if not neighbours:
            node_ok = False
            node_issues.append("no_neighbors")

        valid_mask.append(node_ok)

        if node_issues:
            issues.append({
                "node_index": i,
                "coord": (x, y),
                "problems": node_issues
            })

    return valid_mask, issues


# ------------------------------------------------------------
# Helper: estimate forward direction from tail of traced path
# path uses (row, col)
# ------------------------------------------------------------
def _estimate_path_direction_rc(path, lookback=5):
    if path is None or len(path) < 2:
        return np.array([0.0, 0.0], dtype=float)

    j = max(0, len(path) - 1 - lookback)
    p0 = np.array(path[j], dtype=float)
    p1 = np.array(path[-1], dtype=float)

    v = p1 - p0
    n = np.linalg.norm(v)

    if n < 1e-9:
        return np.array([0.0, 0.0], dtype=float)

    return v / n


# ------------------------------------------------------------
# Continue traced edge until it meets another white pixel
# skeleton expected binary: white > 0
# path uses (row, col)
# ------------------------------------------------------------
def continue_traced_edge_until_white(
    skeleton,
    path,
    max_extension=25,
    search_radius=2,
):
    """Extend endpoint traces so open branches can reconnect to nearby structure."""
    if path is None or len(path) < 2:
        return None, path

    h, w = skeleton.shape[:2]

    current = np.array(path[-1], dtype=float)
    direction = _estimate_path_direction_rc(path)

    if np.linalg.norm(direction) < 1e-9:
        return None, path

    out_path = list(path)
    visited = set((int(r), int(c)) for r, c in out_path)

    for _ in range(max_extension):
        current = current + direction
        rr = int(round(current[0]))
        cc = int(round(current[1]))

        if rr < 0 or rr >= h or cc < 0 or cc >= w:
            break

        # search nearby white pixel
        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                r2 = rr + dr
                c2 = cc + dc

                if r2 < 0 or r2 >= h or c2 < 0 or c2 >= w:
                    continue

                if (r2, c2) in visited:
                    continue

                if skeleton[r2, c2] > 0:
                    out_path.append((r2, c2))
                    return (r2, c2), out_path

        if (rr, cc) not in visited:
            out_path.append((rr, cc))
            visited.add((rr, cc))

    return None, out_path


# ------------------------------------------------------------
# Trace all detected endpoints back using edge_detection tools,
# then continue until another white skeleton point is met.
#
# endpoints expected as (x, y)
# returned hit_points are (row, col)
# ------------------------------------------------------------
def trace_endpoints_to_white(
    skeleton,
    endpoints_xy,
    node_positions_rc=None,
    snap_radius=3,
    max_steps=10000,
    max_extension=25,
    search_radius=2,
):
    """Trace outward from endpoints and add the recovered pixels back to the skeleton."""


    if node_positions_rc is None:
        node_positions_rc = []

    recovered = []
    traced_paths = []

    endpoints_rc = [(int(y), int(x)) for (x, y) in endpoints_xy]

    for ep in endpoints_rc:
        nbrs = get_neighbors(ep, skeleton)

        if len(nbrs) == 0:
            continue

        prev = ep
        nxt = nbrs[0]

        try:
            end_node, path = trace_edge(
                skeleton,
                prev,
                nxt,
                node_positions_rc,
                snap_radius=snap_radius,
                max_steps=max_steps
            )
        except Exception:
            continue

        if path is None or len(path) == 0:
            continue

        traced_paths.append(path)

        # If trace_edge already reached something valid
        if end_node is not None:
            recovered.append(end_node)
            continue

        # otherwise continue until white point found
        hit, full_path = continue_traced_edge_until_white(
            skeleton,
            path,
            max_extension=max_extension,
            search_radius=search_radius
        )

        traced_paths[-1] = full_path

        if hit is not None:
            recovered.append(hit)

    return recovered, traced_paths


def get_skeleton_nodes(
    image,
    work_area=None,
    trace_endpoints=False,
    trace_snap_radius=3,
    trace_max_steps=10000,
    trace_max_extension=25,
    trace_search_radius=2,
):
    """Public preprocessing + node-detection entrypoint used by the pipeline."""
    img, skeleton = preprocess_and_skeletonize(image, work_area)
    all_nodes = []

    # existing node detection
    endpoints, junctions, nodes, node_map = find_skeleton_nodes(skeleton)

    # nodes from detector come as (x, y)
    for node in nodes:
        all_nodes.append((node[1], node[0]))   # convert to (row, col)

    # optional endpoint tracing recovery
    if trace_endpoints:
        recovered_nodes, _ = trace_endpoints_to_white(
            skeleton,
            endpoints,                   # endpoints from detector (x, y)
            node_positions_rc=all_nodes,
            snap_radius=trace_snap_radius,
            max_steps=trace_max_steps,
            max_extension=trace_max_extension,
            search_radius=trace_search_radius
        )

        for pt in recovered_nodes:
            all_nodes.append(pt)

    # cluster nodes
    nodes, cl_node_map = cluster_dbscan(all_nodes)

    node_map = build_node_map_index(nodes)
    for idx, pt in enumerate(nodes, start=1):
        node_map[pt] = idx

    return skeleton, nodes, node_map, cl_node_map
