import numpy as np
from collections import defaultdict
import cv2

"""Trace graph edges between recovered mesh nodes on the skeleton image."""

# 8-connected neighborhood (row, col)
NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

# ----------------------------
# Utility functions
# ----------------------------
def in_bounds(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def get_neighbors(p, skeleton):
    """Return valid skeleton neighbors (row, col)."""
    neighbors = []
    for dr, dc in NEIGHBORS_8:
        q = (p[0] + dr, p[1] + dc)
        if in_bounds(q, skeleton.shape) and skeleton[q] > 0:
            neighbors.append(q)
    return neighbors


def normalize_nodes(nodes):
    """Ensure all nodes are integer tuples (row, col)."""
    return [tuple(map(int, n)) for n in nodes]


def build_node_map(nodes):
    """Map a node coordinate back to its integer node index."""
    return {n: i for i, n in enumerate(nodes)}

def build_node_map_index(nodes):
    """Inverse node map used by later face extraction stages."""
    return {i: n for i, n in enumerate(nodes)}


def snap_to_node(p, node_array, radius=1):
    """
    Safe snapping: only returns node if very close AND exact skeleton alignment likely.
    """
    d2 = (node_array[:, 0] - p[0])**2 + (node_array[:, 1] - p[1])**2
    idx = np.argmin(d2)

    if d2[idx] <= radius**2:
        return tuple(node_array[idx])

    return None


# ----------------------------
# Edge tracing
# ----------------------------
def trace_edge(start_node, start_pixel, skeleton,
               node_map, node_array,
               visited_pixels,
               snap_radius=1,
               max_steps=10000):

    # Follow one skeleton branch until it reaches another known node.
    path = [start_pixel]
    current = start_pixel
    prev = start_node

    local_visited = set([start_node, start_pixel])

    for _ in range(max_steps):

        visited_pixels.add(current)

        # --- snap current ---
        snapped = snap_to_node(current, node_array, snap_radius)
        if snapped is not None and snapped != start_node:
            if snapped in node_map:
                return path, snapped

        neighbors = get_neighbors(current, skeleton)

        # remove previous + already visited
        neighbors = [n for n in neighbors if n != prev and n not in local_visited]

        if len(neighbors) == 0:
            return None

        # --- choose next pixel ---
        if len(neighbors) == 1:
            next_pixel = neighbors[0]
        else:
            def vec(a, b):
                return (b[0] - a[0], b[1] - a[1])

            prev_vec = vec(prev, current)

            def score(n):
                v = vec(current, n)
                # prioritize straight direction, discourage diagonal jumps
                return (v[0]*prev_vec[0] + v[1]*prev_vec[1],
                        -abs(v[0]) - abs(v[1]))

            next_pixel = max(neighbors, key=score)

        # --- snap next ---
        snapped = snap_to_node(next_pixel, node_array, snap_radius)
        if snapped is not None and snapped != start_node:
            if snapped in node_map:
                return path, snapped

        # exact node match
        if next_pixel in node_map and next_pixel != start_node:
            return path, next_pixel

        path.append(next_pixel)
        local_visited.add(next_pixel)

        prev = current
        current = next_pixel

    return None


# ----------------------------
# Edge extraction
# ----------------------------
def extract_edges(skeleton, nodes, min_length=2):
    """Trace all unique skeleton branches between the supplied graph nodes."""
    nodes = normalize_nodes(nodes)
    node_map = build_node_map(nodes)

    edges = set()
    edge_paths = []

    visited_pixels = set()
    visited_starts = set()

    node_array = np.array(nodes)

    for node in nodes:

        neighbors = get_neighbors(node, skeleton)

        for n in neighbors:

            # avoid retracing same direction
            if (node, n) in visited_starts:
                continue

            if skeleton[n] == 0:
                continue

            result = trace_edge(
                node, n,
                skeleton,
                node_map,
                node_array,
                visited_pixels,
                snap_radius=3
            )

            visited_starts.add((node, n))

            if result is None:
                continue

            path, end_node = result

            if end_node not in node_map:
                continue

            i = node_map[node]
            j = node_map[end_node]

            if i == j:
                continue

            if len(path) < min_length:
                continue

            edge = tuple(sorted((i, j)))

            if edge not in edges:
                edges.add(edge)
                edge_paths.append((edge, path))

    return nodes, list(edges), edge_paths


# ----------------------------
# Graph utilities
# ----------------------------
def compute_degrees(edges, num_nodes):
    """Compute simple node degrees from the undirected edge list."""
    degree = [0] * num_nodes
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    return degree


def build_adjacency(edges, num_nodes):
    """Convert an edge list into adjacency lists for later graph algorithms."""
    adj = defaultdict(list)
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    return adj


# ----------------------------
# Main API
# ----------------------------
def build_graph_from_skeleton(skeleton, nodes):
    """
    nodes MUST be in (row, col) format.

    Returns:
    - nodes: list of (row, col)
    - edges: list of (node_i, node_j)
    - adjacency: dict
    - edge_paths: list of ((i,j), path_pixels)
    """
    # This is the main graph-construction step used by the rest of the pipeline.
    nodes, edges, edge_paths = extract_edges(skeleton, nodes)
    adjacency = build_adjacency(edges, len(nodes))

    return {
        "nodes": nodes,
        "node_map": build_node_map(nodes),
        "edges": edges,
        "adjacency": adjacency,
        "edge_paths": edge_paths
    }



