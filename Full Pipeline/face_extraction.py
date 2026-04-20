"""Extract triangle and quadrilateral faces from the recovered graph."""

import numpy as np
from collections import defaultdict


# ============================================================
# Coordinate helpers
# ============================================================

def _to_cartesian_rc(p):
    """
    Convert image-style coordinates (row, col) to Cartesian-like (x, y)
    for angle and area computations.

    input  p = (row, col)
    output    (x=col, y=-row)

    This matches the node convention currently passed from pipeline.py.
    """
    r, c = p
    return np.array([float(c), float(-r)], dtype=float)


def _signed_area_cart(coords):
    """
    Signed polygon area in Cartesian coordinates.
    coords: list of (x, y)
    """
    if len(coords) < 3:
        return 0.0

    x = np.array([p[0] for p in coords], dtype=float)
    y = np.array([p[1] for p in coords], dtype=float)

    return 0.5 * (
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )


def polygon_area(coords):
    """
    Absolute polygon area from raw node coordinates ``(row, col)``.
    """
    cart = [_to_cartesian_rc(p) for p in coords]
    return abs(_signed_area_cart(cart))


# ============================================================
# Geometry helpers
# ============================================================

def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _is_almost_collinear(p0, p1, p2, tol=1e-9):
    """
    Collinearity in Cartesian space.
    """
    a = _to_cartesian_rc(p1) - _to_cartesian_rc(p0)
    b = _to_cartesian_rc(p2) - _to_cartesian_rc(p1)
    return abs(_cross2(a, b)) <= tol


def _segment_intersect_proper(a, b, c, d, tol=1e-9):
    """
    Proper segment intersection test in Cartesian space.
    Ignores touching at shared endpoints.
    """
    a = _to_cartesian_rc(a)
    b = _to_cartesian_rc(b)
    c = _to_cartesian_rc(c)
    d = _to_cartesian_rc(d)

    def orient(p, q, r):
        return _cross2(q - p, r - p)

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    return (o1 * o2 < -tol) and (o3 * o4 < -tol)


def _is_simple_cycle(face, nodes):
    """
    Reject repeated vertices and self-intersections.
    """
    if len(face) < 3:
        return False

    if len(set(face)) != len(face):
        return False

    pts = [nodes[i] for i in face]
    n = len(pts)

    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]

        for j in range(i + 1, n):
            # adjacent edges or same edge
            if j == i:
                continue
            if (j == i + 1) or ((i == 0) and (j == n - 1)):
                continue

            c = pts[j]
            d = pts[(j + 1) % n]

            if _segment_intersect_proper(a, b, c, d):
                return False

    return True


def is_convex(coords, tol=1e-9):
    """
    Convexity test on raw node coords (row, col).
    Collinear turns are allowed.
    """
    pts = [_to_cartesian_rc(p) for p in coords]
    n = len(pts)

    if n < 4:
        return True

    sign = 0
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]

        z = _cross2(p1 - p0, p2 - p1)

        if abs(z) <= tol:
            continue

        s = 1 if z > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False

    return True


# ============================================================
# Neighbor ordering
# ============================================================

def sort_neighbors_ccw(nodes, adjacency):
    """
    Sort each node's neighbors counter-clockwise so face walks are consistent.
    """
    ordered = {}

    for u, pu in nodes.items():
        cu = _to_cartesian_rc(pu)
        neighbors = list(adjacency[u])

        def angle(v):
            cv = _to_cartesian_rc(nodes[v])
            d = cv - cu
            return np.arctan2(d[1], d[0])

        neighbors.sort(key=angle)
        ordered[u] = neighbors

    return ordered


def next_ccw(u, v, ordered_neighbors):
    """
    Given a directed half-edge ``u -> v``, choose the next edge around the face.
    """
    nbrs = ordered_neighbors[v]
    idx = nbrs.index(u)
    w = nbrs[idx - 1]
    return w


# ============================================================
# Cycle normalization / simplification
# ============================================================

def _rotate_to_smallest(seq):
    m = min(seq)
    k = seq.index(m)
    return seq[k:] + seq[:k]


def _canonical_face(face):
    """
    Canonical representation up to rotation and reversal.
    """
    f1 = _rotate_to_smallest(face[:])
    f2 = _rotate_to_smallest(face[::-1])
    return tuple(f1) if tuple(f1) < tuple(f2) else tuple(f2)


def _remove_consecutive_duplicates(face):
    out = []
    for x in face:
        if not out or out[-1] != x:
            out.append(x)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _simplify_collinear_vertices(face, nodes):
    """
    Remove vertices that lie on a straight boundary segment.
    This is important when the graph contains degree-2 chain points.
    """
    face = _remove_consecutive_duplicates(face)

    changed = True
    while changed and len(face) >= 3:
        changed = False
        new_face = []
        n = len(face)

        for i in range(n):
            a = face[(i - 1) % n]
            b = face[i]
            c = face[(i + 1) % n]

            if _is_almost_collinear(nodes[a], nodes[b], nodes[c]):
                changed = True
                continue

            new_face.append(b)

        if len(new_face) < 3:
            return []

        face = _remove_consecutive_duplicates(new_face)

    return face


# ============================================================
# Face extraction
# ============================================================

def extract_faces(nodes, edges):
    """
    Walk graph half-edges to enumerate simple candidate faces.
    """
    adjacency = defaultdict(set)
    undirected_edges = set()

    for u, v in edges:
        if u == v:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)
        undirected_edges.add(tuple(sorted((u, v))))

    ordered_neighbors = sort_neighbors_ccw(nodes, adjacency)

    visited_half_edges = set()
    faces = []
    seen_faces = set()

    for u, v in undirected_edges:
        for start in [(u, v), (v, u)]:
            if start in visited_half_edges:
                continue

            face = []
            curr = start
            max_steps = max(10, 4 * len(undirected_edges))

            for _ in range(max_steps):
                a, b = curr

                if curr in visited_half_edges:
                    # if we returned to start, fine; otherwise bad walk
                    if curr == start:
                        break
                    face = []
                    break

                visited_half_edges.add(curr)
                face.append(a)

                if b not in ordered_neighbors or len(ordered_neighbors[b]) < 2:
                    face = []
                    break

                c = next_ccw(a, b, ordered_neighbors)
                curr = (b, c)

                if curr == start:
                    break
            else:
                face = []

            if len(face) < 3:
                continue

            # Simplify away degree-2 chain points before classifying the polygon.
            face = _remove_consecutive_duplicates(face)
            face = _simplify_collinear_vertices(face, nodes)

            if len(face) < 3:
                continue

            if not _is_simple_cycle(face, nodes):
                continue

            key = _canonical_face(face)
            if key in seen_faces:
                continue
            seen_faces.add(key)

            faces.append(face)

    return faces


# ============================================================
# Filter / classify faces
# ============================================================

def classify_faces(nodes, faces):
    """
    Keep only interior triangular and convex quadrilateral faces.
    """
    if not faces:
        return [], []

    face_data = []

    for f in faces:
        coords = [nodes[i] for i in f]
        cart = [_to_cartesian_rc(p) for p in coords]
        signed_area = _signed_area_cart(cart)
        area = abs(signed_area)

        if area < 1e-9:
            continue

        face_data.append((f, signed_area, area))

    if not face_data:
        return [], []

    # Remove the outer infinite face:
    # with this walk convention, the outer face is the unique face
    # with the largest absolute area.
    outer_idx = max(range(len(face_data)), key=lambda i: face_data[i][2])
    face_data.pop(outer_idx)

    triangles = []
    quads = []

    for f, signed_area, area in face_data:
        coords = [nodes[i] for i in f]

        if len(f) == 3:
            triangles.append(f)

        elif len(f) == 4:
            if is_convex(coords):
                quads.append(f)

    return triangles, quads


# ============================================================
# Main API
# ============================================================

def extract_mesh_elements(nodes, edges):
    """
    Public API: return the triangle and quadrilateral mesh elements.
    """
    faces = extract_faces(nodes, edges)
    triangles, quads = classify_faces(nodes, faces)
    return triangles, quads
