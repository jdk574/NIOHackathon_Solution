"""Recover missed nodes and edges from the leftover skeleton after the first graph pass."""

import numpy as np
import cv2

from edge_detection import build_graph_from_skeleton, normalize_nodes, build_node_map, build_node_map_index
from node_extraction import find_skeleton_nodes, cluster_dbscan, snap_points_bruteforce
from nodes_classification import classify_all_nodes


def _unique_tuples(seq):
    """Deduplicate coordinate tuples while keeping their first-seen order."""
    seen = set()
    out = []
    for p in seq:
        p = tuple(map(int, p))
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _dist2(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def _find_matching_old_node(new_node_rc, old_nodes_rc, radius):
    """
    new_node_rc, old_nodes_rc in (row, col)
    Returns matched old node index or None.
    """
    if len(old_nodes_rc) == 0:
        return None

    r2 = radius * radius
    best_idx = None
    best_d2 = None

    for i, old_node in enumerate(old_nodes_rc):
        d2 = _dist2(new_node_rc, old_node)
        if d2 <= r2 and (best_d2 is None or d2 < best_d2):
            best_d2 = d2
            best_idx = i

    return best_idx


def circular_mask(shape, center, radius):
    """
    Create a binary circular mask in image coordinates.
    """
    h, w = shape
    cy, cx = center
    Y, X = np.ogrid[:h, :w]
    mask = (Y - cy) ** 2 + (X - cx) ** 2 <= radius ** 2
    return mask.astype(np.uint8)


def remove_edge_paths_from_skeleton(
    skeleton,
    edge_paths,
    old_nodes=None,
    remove_end_neighborhood=0,
    old_node_mask_radius=0,
):
    """
    Delete traced edge pixels from a skeleton copy, and optionally also mask
    a circular neighborhood around previously accepted old nodes.

    Parameters
    ----------
    skeleton : np.ndarray
        Binary skeleton, 0/1 or 0/255
    edge_paths : list of ((i, j), path_pixels)
        path_pixels expected in (row, col)
    old_nodes : list[(row, col)] or None
        Existing accepted nodes to mask around in residual construction
    remove_end_neighborhood : int
        Optional tiny cleanup radius around removed edge pixels
    old_node_mask_radius : int
        Radius of circular masking around each old node

    Returns
    -------
    residual : np.ndarray
        Binary residual skeleton in 0/1
    """
    residual = (skeleton > 0).astype(np.uint8).copy()
    h, w = residual.shape

    # 1) remove already-traced edge pixels
    for _, path in edge_paths:
        if not path:
            continue

        for p in path:
            y, x = map(int, p)
            if 0 <= y < h and 0 <= x < w:
                residual[y, x] = 0

                if remove_end_neighborhood > 0:
                    y0 = max(0, y - remove_end_neighborhood)
                    y1 = min(h, y + remove_end_neighborhood + 1)
                    x0 = max(0, x - remove_end_neighborhood)
                    x1 = min(w, x + remove_end_neighborhood + 1)

                    for yy in range(y0, y1):
                        for xx in range(x0, x1):
                            if (yy - y) ** 2 + (xx - x) ** 2 <= remove_end_neighborhood ** 2:
                                residual[yy, xx] = 0

    # 2) mask around old nodes so residual extraction does not rediscover them
    if old_nodes is not None and old_node_mask_radius > 0:
        for node in old_nodes:
            y, x = map(int, node)
            mask = circular_mask((h, w), (y, x), old_node_mask_radius)
            residual[mask > 0] = 0

    return residual


def detect_nodes_on_residual(
    residual_skeleton,
    classify_candidates=True,
    classify_kwargs=None,
):
    """
    Detect candidate nodes on the unexplained residual skeleton.

    Returns
    -------
    residual_nodes_rc : list[(row, col)]
    """
    if classify_kwargs is None:
        classify_kwargs = {}

    skel = (residual_skeleton > 0).astype(np.uint8)

    # find_skeleton_nodes returns (x, y)
    endpoints, junctions, all_nodes_xy, _ = find_skeleton_nodes(skel)

    if len(all_nodes_xy) == 0:
        print("No residual nodes found")
        return []


    snapped_xy = all_nodes_xy

    # convert (x, y) -> (row, col)
    residual_nodes_rc = [(int(y), int(x)) for x, y in snapped_xy]
    residual_nodes_rc = _unique_tuples(residual_nodes_rc)

    if classify_candidates and len(residual_nodes_rc) > 0:
        confirmed, rejected, _ = classify_all_nodes(
            skel,
            residual_nodes_rc,
            **classify_kwargs
        )
        residual_nodes_rc = _unique_tuples(confirmed)

    print(f"Residual nodes: {len(residual_nodes_rc)}")
    return residual_nodes_rc


def merge_residual_graph_into_original(
    old_nodes,
    old_edges,
    old_edge_paths,
    residual_graph,
    match_radius=6,
):
    """
    Merge a small residual graph back into the original graph.

    Parameters
    ----------
    old_nodes : list[(row, col)]
    old_edges : list[(i, j)]
    old_edge_paths : list[((i, j), path)]
    residual_graph : dict
        output of build_graph_from_skeleton on the residual skeleton
    match_radius : int
        If a new residual node is within this radius of an old node,
        use the old node instead of adding a new one.

    Returns
    -------
    merged_nodes, merged_edges, merged_edge_paths, mapping_info
    """
    old_nodes = normalize_nodes(old_nodes)
    merged_nodes = list(old_nodes)

    existing_edges = {tuple(sorted(e)) for e in old_edges}
    merged_edges = list(existing_edges)
    merged_edge_paths = list(old_edge_paths)

    residual_nodes = normalize_nodes(residual_graph["nodes"])
    residual_edges = residual_graph["edges"]
    residual_edge_paths = residual_graph["edge_paths"]

    print(f"Length of residual edges: {len(residual_edges)}")

    # map residual node index -> merged node index
    residual_to_merged = {}

    for r_idx, r_node in enumerate(residual_nodes):
        match_idx = _find_matching_old_node(r_node, old_nodes, match_radius)

        if match_idx is not None:
            residual_to_merged[r_idx] = match_idx
        else:
            residual_to_merged[r_idx] = len(merged_nodes)
            merged_nodes.append(r_node)

    # remap residual edges into merged graph
    new_edges_added = 0
    for edge in residual_edges:
        i_old, j_old = edge
        i_new = residual_to_merged[i_old]
        j_new = residual_to_merged[j_old]
        print(f"Old, new indices: {i_old}, {j_old}, {i_new}, {j_new}")

        if i_new == j_new:
            continue

        merged_edge = tuple(sorted((i_new, j_new)))
        print(merged_edge not in existing_edges)
        if merged_edge not in existing_edges:
            existing_edges.add(merged_edge)
            merged_edges.append(merged_edge)
            new_edges_added += 1
            print(f"Added edge: {merged_edge}")

    # remap residual edge_paths too
    existing_edge_paths_keys = {tuple(sorted(e)) for e, _ in merged_edge_paths}

    for edge, path in residual_edge_paths:
        i_old, j_old = edge
        i_new = residual_to_merged[i_old]
        j_new = residual_to_merged[j_old]

        if i_new == j_new:
            continue

        merged_edge = tuple(sorted((i_new, j_new)))
        if merged_edge not in existing_edge_paths_keys:
            merged_edge_paths.append((merged_edge, path))
            existing_edge_paths_keys.add(merged_edge)

    mapping_info = {
        "residual_to_merged": residual_to_merged,
        "num_old_nodes": len(old_nodes),
        "num_merged_nodes": len(merged_nodes),
        "num_new_nodes_added": len(merged_nodes) - len(old_nodes),
        "num_new_edges_added": new_edges_added,
    }

    return merged_nodes, merged_edges, merged_edge_paths, mapping_info


def recover_leftover_graph(
    skeleton,
    old_nodes,
    old_edges,
    old_edge_paths,
    match_radius=6,
    classify_candidates=True,
    classify_kwargs=None,
    remove_end_neighborhood=0,
    old_node_mask_radius=0,
):
    """
    Run the full residual-recovery pass used after the initial graph extraction.
    """
    if classify_kwargs is None:
        classify_kwargs = {
            "outer_radius": 20,
            "inner_radius": 4,
            "direction_step": 5,
            "angle_tol_deg": 15,
            "min_branch_length": 3,
            "min_branches_for_node": 2,
            "round_input_nodes": True,
            "return_debug": False,
        }

    skel = (skeleton > 0).astype(np.uint8)

    # Strip already-explained structure so only missed branches remain.
    residual_skeleton = remove_edge_paths_from_skeleton(
        skel,
        old_edge_paths,
        old_nodes=old_nodes,
        remove_end_neighborhood=remove_end_neighborhood,
        old_node_mask_radius=old_node_mask_radius,
    )

    # Search the residual skeleton for node candidates and rebuild a small graph.
    residual_nodes = detect_nodes_on_residual(
        residual_skeleton,
        classify_candidates=classify_candidates,
        classify_kwargs=classify_kwargs,
    )

    if len(residual_nodes) > 0:
        residual_graph = build_graph_from_skeleton(residual_skeleton, residual_nodes)
    else:
        residual_graph = {
            "nodes": [],
            "node_map": {},
            "edges": [],
            "adjacency": {},
            "edge_paths": [],
        }

    # Finally fold residual nodes/edges back into the main graph, matching nearby nodes.
    merged_nodes, merged_edges, merged_edge_paths, mapping_info = merge_residual_graph_into_original(
        old_nodes=old_nodes,
        old_edges=old_edges,
        old_edge_paths=old_edge_paths,
        residual_graph=residual_graph,
        match_radius=match_radius,
    )

    return {
        "residual_skeleton": residual_skeleton,
        "residual_nodes": residual_nodes,
        "residual_graph": residual_graph,
        "merged_nodes": merged_nodes,
        "merged_edges": merged_edges,
        "merged_edge_paths": merged_edge_paths,
        "merged_node_map": build_node_map_index(merged_nodes),
        "mapping_info": mapping_info,
    }
