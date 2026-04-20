import numpy as np
from edge_detection import *
from utils import *
from sklearn.cluster import DBSCAN

"""Detect strong direction changes so long traced paths can be split into corners."""

import numpy as np


def _normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def _direction_at(path_arr, i, window):
    """
    path_arr: Nx2 array in (row, col)
    returns local tangent direction at index i using a symmetric/asymmetric window
    """
    n = len(path_arr)
    left = max(0, i - window)
    right = min(n - 1, i + window)

    if right <= left:
        return None

    v = path_arr[right] - path_arr[left]
    return _normalize(v)


def _turn_angle(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot)


def _merge_peaks(peaks, min_index_gap=6):
    """
    Merge nearby turn detections along the same path.
    Keeps the strongest one in each local group.
    peaks: list of dicts with key 'path_index' and 'angle_deg'
    """
    if not peaks:
        return []

    peaks = sorted(peaks, key=lambda d: d["path_index"])
    merged = []
    group = [peaks[0]]

    for p in peaks[1:]:
        if p["path_index"] - group[-1]["path_index"] <= min_index_gap:
            group.append(p)
        else:
            best = max(group, key=lambda d: d["angle_deg"])
            merged.append(best)
            group = [p]

    best = max(group, key=lambda d: d["angle_deg"])
    merged.append(best)
    return merged


def edge_global_turns(
    path,
    direction_window=12,
    comparison_step=8,
    angle_threshold_deg=25.0,
    min_peak_gap=6,
    ignore_ends=4
):
    """Measure angle changes along one traced edge path."""
    """
    Detect global turn points on one already-traced edge path.

    Parameters
    ----------
    path : list of (row, col)
        Ordered pixels along the edge.
    direction_window : int
        Size of the tangent-estimation window.
        Larger = more global, less sensitive to noise.
    comparison_step : int
        Compare direction at i-step vs i+step.
        Larger = more global turn detection.
    angle_threshold_deg : float
        Minimum turn angle to mark as a bend.
    min_peak_gap : int
        Merge nearby detections closer than this many path pixels.
    ignore_ends : int
        Ignore a few pixels at each end of the path.

    Returns
    -------
    result : dict
        {
            "has_turn": bool,
            "max_angle_deg": float,
            "turn_points": [
                {
                    "path_index": int,
                    "pixel_rc": (row, col),
                    "angle_deg": float
                }, ...
            ]
        }
    """
    if path is None or len(path) < 2 * direction_window + comparison_step + 2:
        return {
            "has_turn": False,
            "max_angle_deg": 0.0,
            "turn_points": []
        }

    path_arr = np.asarray(path, dtype=np.float32)

    start = max(ignore_ends, direction_window)
    stop = len(path_arr) - max(ignore_ends, direction_window) - comparison_step

    if stop <= start:
        return {
            "has_turn": False,
            "max_angle_deg": 0.0,
            "turn_points": []
        }

    raw_peaks = []
    max_angle_deg = 0.0

    for i in range(start, stop):
        v1 = _direction_at(path_arr, i, direction_window)
        v2 = _direction_at(path_arr, i + comparison_step, direction_window)

        angle_rad = _turn_angle(v1, v2)
        angle_deg = float(np.degrees(angle_rad))

        if angle_deg > max_angle_deg:
            max_angle_deg = angle_deg

        if angle_deg >= angle_threshold_deg:
            r, c = path[i]
            raw_peaks.append({
                "path_index": i,
                "pixel_rc": (int(r), int(c)),
                "angle_deg": angle_deg
            })

    turn_points = _merge_peaks(raw_peaks, min_index_gap=min_peak_gap)

    return {
        "has_turn": len(turn_points) > 0,
        "max_angle_deg": max_angle_deg,
        "turn_points": turn_points
    }


def check_edges_global_curvature(
    edge_paths,
    direction_window=8,
    comparison_step=4,
    angle_threshold_deg=10.0,
    min_peak_gap=6,
    ignore_ends=4
):
    """Run turn detection over every traced edge in the current graph."""
    """
    Run global-curvature detection on all traced edges.

    Parameters
    ----------
    edge_paths : list of ((i, j), path)
        Output like your graph["edge_paths"].

    Returns
    -------
    results : list of dict
        One entry per edge:
        {
            "edge": (i, j),
            "has_turn": bool,
            "max_angle_deg": float,
            "turn_points": [...]
        }
    """
    results = []

    for edge, path in edge_paths:
        res = edge_global_turns(
            path=path,
            direction_window=direction_window,
            comparison_step=comparison_step,
            angle_threshold_deg=angle_threshold_deg,
            min_peak_gap=min_peak_gap,
            ignore_ends=ignore_ends
        )
        res["edge"] = edge
        results.append(res)

    return results

def split_edges_on_turns(
    graph,
    curvature_results,
    merge_dist=15,
    dbscan_eps=8,
    dbscan_min_samples=2,
    snap_path_endpoints=True
):
    """Insert new nodes at detected turn locations and rebuild the edge list."""

    """

    Split graph edges at detected turn points, then merge nearby nodes with DBSCAN,

    and finally snap edges to the kept nodes.

    IMPORTANT

    ---------

    This function keeps the graph coordinate convention as (row, col).

    Parameters

    ----------

    graph : dict

        {

            "nodes": [(row, col), ...],

            "edges": [(i, j), ...],

            "edge_paths": [((i, j), path), ...]

        }

    curvature_results : list[dict]

        Output of check_edges_global_curvature()

    merge_dist : float

        Distance threshold used while inserting intermediate turn nodes before DBSCAN.

    dbscan_eps : float

        DBSCAN epsilon for post-split node merging.

    dbscan_min_samples : int

        DBSCAN min_samples for post-split node merging.

    snap_path_endpoints : bool

        If True, force each path to start/end exactly on its kept node coordinates.

    Returns

    -------

    updated_graph : dict

    """

    nodes = [tuple(map(int, n)) for n in graph["nodes"]]

    original_edge_paths = {

        tuple(sorted(e)): p for e, p in graph["edge_paths"]

    }

    split_edges = []

    split_edge_paths = []

    def dist(a, b):

        a = np.asarray(a, dtype=float)

        b = np.asarray(b, dtype=float)

        return np.linalg.norm(a - b)

    def find_or_add_node(pt_rc):

        """

        pt_rc must be (row, col).

        Merge immediately with an existing node if close enough.

        """

        pt_rc = tuple(map(int, pt_rc))

        for idx, n in enumerate(nodes):

            if dist(n, pt_rc) < merge_dist:

                return idx

        nodes.append(pt_rc)

        return len(nodes) - 1

    # --------------------------------------------------

    # 1) Split edges at turn points

    # --------------------------------------------------

    for res in curvature_results:

        i, j = tuple(res["edge"])

        edge_key = tuple(sorted((i, j)))

        if edge_key not in original_edge_paths:

            continue

        path = original_edge_paths[edge_key]

        if not res["has_turn"] or len(res["turn_points"]) == 0:

            split_edges.append(edge_key)

            split_edge_paths.append((edge_key, path))

            continue

        turn_points = sorted(res["turn_points"], key=lambda d: d["path_index"])

        # keep only interior unique split positions

        split_indices = []

        for tp in turn_points:

            idx = int(tp["path_index"])

            if 0 < idx < len(path) - 1:

                if not split_indices or idx != split_indices[-1]:

                    split_indices.append(idx)

        if len(split_indices) == 0:

            split_edges.append(edge_key)

            split_edge_paths.append((edge_key, path))

            continue

        indices = [0] + split_indices + [len(path) - 1]

        current_start_node = i

        for k in range(len(indices) - 1):

            start_idx = indices[k]

            end_idx = indices[k + 1]

            sub_path = list(path[start_idx:end_idx + 1])

            if len(sub_path) < 2:

                continue

            if k < len(split_indices):

                # intermediate turn node; keep graph convention (row, col)

                turn_pt_rc = tuple(map(int, path[end_idx]))

                new_node = find_or_add_node(turn_pt_rc)

                if current_start_node != new_node:

                    seg_edge = tuple(sorted((current_start_node, new_node)))

                    split_edges.append(seg_edge)

                    split_edge_paths.append((seg_edge, sub_path))

                current_start_node = new_node

            else:

                if current_start_node != j:

                    seg_edge = tuple(sorted((current_start_node, j)))

                    split_edges.append(seg_edge)

                    split_edge_paths.append((seg_edge, sub_path))

    # --------------------------------------------------

    # 2) DBSCAN cluster all nodes after splitting

    #    Same spirit as earlier pipeline:

    #    - merge true clusters

    #    - keep noise as single nodes

    # --------------------------------------------------

    def cluster_dbscan_rc(points, eps=6, min_samples=3):

        if len(points) == 0:

            return [], {}, {}

        pts = np.asarray(points, dtype=float)

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)

        labels = clustering.labels_

        clustered_nodes = []

        point_to_cluster = {}

        cluster_coord_to_index = {}

        # true clusters first

        valid_labels = sorted(lab for lab in set(labels) if lab != -1)

        for label in valid_labels:

            member_ids = np.where(labels == label)[0]

            cluster_pts = pts[member_ids]

            center = tuple(np.rint(cluster_pts.mean(axis=0)).astype(int))

            if center not in cluster_coord_to_index:

                cluster_coord_to_index[center] = len(clustered_nodes)

                clustered_nodes.append(center)

            new_idx = cluster_coord_to_index[center]

            for old_idx in member_ids:

                point_to_cluster[old_idx] = new_idx

        # keep DBSCAN noise points as singles

        noise_ids = np.where(labels == -1)[0]

        for old_idx in noise_ids:

            coor = tuple(np.rint(pts[old_idx]).astype(int))

            if coor not in cluster_coord_to_index:

                cluster_coord_to_index[coor] = len(clustered_nodes)

                clustered_nodes.append(coor)

            new_idx = cluster_coord_to_index[coor]

            point_to_cluster[old_idx] = new_idx

        return clustered_nodes, point_to_cluster, cluster_coord_to_index

    clustered_nodes, old_to_new_node_idx, _ = cluster_dbscan_rc(

        nodes,

        eps=dbscan_eps,

        min_samples=dbscan_min_samples

    )

    # --------------------------------------------------

    # 3) Remap split edges to clustered node ids

    # --------------------------------------------------

    def snap_path_to_nodes(path, start_rc, end_rc):

        """

        Force path to touch kept nodes at both ends.

        """

        path = list(path)

        if len(path) == 0:

            return [start_rc, end_rc]

        start_rc = tuple(map(int, start_rc))

        end_rc = tuple(map(int, end_rc))

        if path[0] != start_rc:

            path = [start_rc] + path

        if path[-1] != end_rc:

            path = path + [end_rc]

        # remove consecutive duplicates

        cleaned = [path[0]]

        for p in path[1:]:

            if p != cleaned[-1]:

                cleaned.append(p)

        return cleaned

    merged_edge_paths_map = {}

    for (old_i, old_j), path in split_edge_paths:

        new_i = old_to_new_node_idx[old_i]

        new_j = old_to_new_node_idx[old_j]

        if new_i == new_j:

            continue

        edge = tuple(sorted((new_i, new_j)))

        start_rc = clustered_nodes[new_i]

        end_rc = clustered_nodes[new_j]

        if snap_path_endpoints:

            path = snap_path_to_nodes(path, start_rc, end_rc)

        # Keep the longest path if duplicate edges appear after clustering

        if edge not in merged_edge_paths_map:

            merged_edge_paths_map[edge] = path

        else:

            if len(path) > len(merged_edge_paths_map[edge]):

                merged_edge_paths_map[edge] = path

    final_edges = list(merged_edge_paths_map.keys())

    final_edge_paths = [(e, merged_edge_paths_map[e]) for e in final_edges]

    return {
        "nodes": clustered_nodes,
        "edges": final_edges,
        "edge_paths": final_edge_paths,
        "adjacency": build_adjacency(final_edges, len(clustered_nodes)),
        "node_map": build_node_map_index(clustered_nodes)
    }
