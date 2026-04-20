"""Visualization helpers for debugging intermediate graph-extraction stages."""

import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

from utils import open_image

def skeleton_vis(skeleton, image):
    """Overlay a binary skeleton in red on top of the source image."""
    # Make skeleton visible (white lines)
    vis = (skeleton * 255).astype(np.uint8)

    if len(image.shape) == 2:
        # grayscale → convert to BGR
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        # already color → copy directly
        img_color = image.copy()

    # Overlay skeleton in red
    img_color[skeleton == 1] = [0, 0, 255]

    return img_color


def draw_nodes(image, endpoints, junctions=None):
    """Draw endpoint and junction detections for quick visual checks."""
    # Handle both grayscale and color images
    if len(image.shape) == 2:
        # grayscale → convert to BGR
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        # already color → copy directly
        img_color = image.copy()


    for x, y in endpoints:
        cv2.circle(img_color, (x, y), 3, (255, 0, 0), -1)  # blue

    if junctions is not None:
        for x, y in junctions:
            cv2.circle(img_color, (x, y), 4, (0, 0, 255), -1)  # red

    return img_color

def visualize_graph(skeleton, nodes, edge_paths, scale=2, show_ids=False):
    """
    Visualize the traced graph on top of a skeleton image.
    """

    # Convert skeleton to RGB image
    img = (skeleton * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Helper to scale points
    def scale_pt(p):
        return (int(p[1] * scale), int(p[0] * scale))  # (x,y) for OpenCV

    # Give each edge path its own color so tracing mistakes are easy to spot.
    for (i, j), path in edge_paths:
        color = [random.randint(50, 255) for _ in range(3)]

        for p in path:
            cv2.circle(img, scale_pt(p), radius=3, color=color, thickness=-1)

    # Draw nodes after edges so the key points stay visible.
    for idx, p in enumerate(nodes):
        cv2.circle(img, scale_pt(p), radius=3, color=(0, 0, 255), thickness=-1)

        if show_ids:
            cv2.putText(
                img,
                str(idx),
                scale_pt(p),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

    return img

def draw_circles(image, centers, radius=5, color=(0, 255, 0), thickness=1):
    """Draw circles for centers given in ``(x, y)`` convention."""
    img = open_image(image)
    for (x, y) in centers:
        cv2.circle(img, (int(x), int(y)), radius, color, thickness)

    return img

def visualize_faces(nodes, triangles, quads, title):
    """Display extracted triangles and quads using a Cartesian-style plot."""
    fig, ax = plt.subplots()

    def transform(p):
        x, y = p
        return y, -x

    for tri in triangles:
        pts = [transform(nodes[i]) for i in tri]
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]

        ax.fill(xs, ys, color='g', alpha=0.4)
        ax.plot(xs, ys)

    for quad in quads:
        pts = [transform(nodes[i]) for i in quad]
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]

        ax.fill(xs, ys, color='r', alpha=0.4)
        ax.plot(xs, ys)

    for i, (x, y) in nodes.items():
        tx, ty = transform((x, y))
        ax.plot(tx, ty, 'ko')
        ax.text(tx, ty, str(i), fontsize=8)

    fig.suptitle(title)
    ax.set_aspect('equal')
    plt.show()
